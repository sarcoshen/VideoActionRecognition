import os
import random
import numpy as np
import torch
import torch.utils.data
import csv
import cv2

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
# import slowfast.utils.logging as logging


def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
    # print(size, 'QQQ')

    frame_data = []
    while True:
        rect, frame = cap.read()
        if rect:
            # print(frame.shape, '0000000000')
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_data.append(frame)
        else:
            break
    return frame_data, size



def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def cv2_decode(video_container, sampling_rate=2, num_frames=32, clip_idx=-1, num_clips=10, target_fps=30):
    fps = float(video_container.get(cv2.CAP_PROP_FPS))
    frames_length = int(video_container.get(cv2.CAP_PROP_FRAME_COUNT))

    # 这里应该是获取了起始帧和结束帧
    start_idx, end_idx = get_start_end_idx(frames_length, sampling_rate * num_frames / target_fps * fps, clip_idx,
                                           num_clips)

    decode_all_video = False
    frame_data, size = get_frames(video_container)
    video_frames = frame_data[start_idx: end_idx + 1]    # 这里需要确认一下是不是需要加1
    max_pts = end_idx

    frames = [frame.to_ndarray() for frame in video_frames]
    frames = torch.as_tensor(np.stack(frames))            # 这里可能有问题
    return frames, fps, decode_all_video, start_idx, end_idx



def tianchi_decode(video_container, sampling_rate=2, num_frames=32, clip_idx=-1, num_clips=10, video_meta=None, target_fps=30,):
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
    try:
        frames, fps, decode_all_video, start_idx, end_idx = cv2_decode(video_container, sampling_rate, num_frames, clip_idx, num_clips, target_fps)
    except Exception as e:
        print("Failed to decode with pyav with exception: {}".format(e))
        return None

    if frames is None:
        return frames


def process_short_video(frame_data, seq_len=64):
    """
    当视频的长度不足64帧时，按照均匀分布进行上采样
    :param frame_data: 帧列表
    :param seq_len: 最短的视频长度，默认是64
    :return:
    """
    if len(frame_data) < seq_len:
        # print(len(frame_data), '+++++++')
        num = seq_len - len(frame_data)# + 1
        indices = np.linspace(start=0, stop=len(frame_data)-1, num=num, dtype='int32')
        ind = np.arange(len(frame_data)).astype('int32')
        indices = list(np.sort(np.append(indices, ind)))

        frames = []
        # print(frame_data, '&&&&&&&&')
        for i in range(len(indices)):
            # print(indices)
            # print(i, indices[i], len(frame_data))
            # print(frame_data[indices[i]].shape)
            frames.append(frame_data[indices[i]])
    else:
        frames = frame_data
        indices = []
    return frames, indices


class Kinetics(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, num_retries=10):
        assert mode in ["train", "val", "test",], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.seq_len = self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE

        self._video_meta = {}
        self._num_retries = num_retries     # 这个参数是干什么的

        if self.mode in ["train", "val"]:
            self._num_clips = 1     # 这个参数是干什么的
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        self._construct_loader()

        self.index_list = []
        self.indexes = []

    def _construct_loader(self):
        # 创建视频加载器
        path_to_file = os.path.join("{}.csv".format(self.mode))      # 获取对应的csv文件的路径
        # path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode))  # 获取对应的csv文件的路径
        assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []          # 用来存储视频的路径
        self._labels = []                  # 用来存储视频的标签
        self._spatial_temporal_idx = []    # 这个不知道， 用来空间抽样

        with open(path_to_file, 'r') as f:
            reader = csv.reader(f)
            for clip_idx, (path, label) in enumerate(reader):
                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, path))      # 这里是得到了所有视频的路径
                    self._labels.append(int(label))                                                 # 得到所有的label
                    self._spatial_temporal_idx.append(idx)                                          # 空间抽样的索引，表示第几段视频，其实没用上
                    self._video_meta[clip_idx * self._num_clips + idx] = {}

    def __getitem__(self, index):
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS)
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS)
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # for _ in range(self._num_retries):     # 尝试多次
        #     frame_data = None
        #     video_container = None
        try:
            # print(self._path_to_videos[index], '*****')
            frame_data, size = get_frames(self._path_to_videos[index])
            # print(len(frame_data), '0000000')


            num_frame_data = len(frame_data)
            if num_frame_data < self.seq_len:
                frame_data, indices = process_short_video(frame_data, self.seq_len)
                start_id = 0
            else:
                start_id = random.randint(0, num_frame_data-self.seq_len)
            frames = frame_data[start_id: start_id + self.seq_len: self.cfg.DATA.SAMPLING_RATE]
            self.indexes.append(index)
        except:
            if len(self.indexes) < 1:
                index = random.randint(0, len(self._path_to_videos))
            else:

                k = random.randint(0, len(self.indexes)-1)
                # print(self.indexes, k)
                # print('from processed')
                index = self.indexes[k]

            frame_data, size = get_frames(self._path_to_videos[index])
            # print(len(frame_data), '0000000')

            num_frame_data = len(frame_data)
            if num_frame_data < self.seq_len:
                frame_data, indices = process_short_video(frame_data, self.seq_len)
                start_id = 0
            else:
                start_id = random.randint(0, num_frame_data - self.seq_len)
            frames = frame_data[start_id: start_id + self.seq_len: self.cfg.DATA.SAMPLING_RATE]

        # ind = torch.linspace(0, len(frames), self.cfg.DATA.NUM_FRAMES).long()
        # frames = frames[ind]

        # 后面的可能不对
        frames = np.array(frames)
        #frames = frames.astype(double)
        frames = torch.tensor(frames)
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # print('frames.shape: {}'.format(frames.shape))
        #
        # frames = self.spatial_sampling(
        #     frames,
        #     spatial_idx=spatial_sample_index,
        #     min_scale=min_scale,
        #     max_scale=max_scale,
        #     crop_size=crop_size,
        # )
        # print(frames.shape)
        label = self._labels[index]
        frames = utils.pack_pathway_output(self.cfg, frames)
        # frames = preprocess_action_data(frames, self.cfg)

        return frames, label, index, {}

    # def __getitem__(self, index):
    #     if self.mode in ["train", "val"]:
    #         # -1 indicates random sampling.
    #         temporal_sample_index = -1
    #         spatial_sample_index = -1
    #         min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
    #         max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
    #         crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
    #     elif self.mode in ["test"]:
    #         temporal_sample_index = (
    #             self._spatial_temporal_idx[index]
    #             // self.cfg.TEST.NUM_SPATIAL_CROPS)
    #         spatial_sample_index = (
    #             self._spatial_temporal_idx[index]
    #             % self.cfg.TEST.NUM_SPATIAL_CROPS)
    #         min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
    #         assert len({min_scale, max_scale, crop_size}) == 1
    #     else:
    #         raise NotImplementedError(
    #             "Does not support {} mode".format(self.mode)
    #         )
    #
    #     # for _ in range(self._num_retries):     # 尝试多次
    #     # frame_data = None
    #     # video_container = None
    #
    #     frame_data, size = get_frames(self._path_to_videos[index])
    #     num_frame_data = len(frame_data)
    #     while num_frame_data == 0:
    #         if len(self.index_list)>0:
    #             ind = random.randint(0, len(self.index_list)-1)
    #             index = self.index_list[ind]
    #         else:
    #             index = random.randint(0, len(self._path_to_videos))
    #
    #         frame_data, size = get_frames(self._path_to_videos[index])
    #         num_frame_data = len(frame_data)
    #
    #     if 0 < num_frame_data < self.seq_len:
    #         frame_data, indices = process_short_video(frame_data, self.seq_len)
    #         start_id = 0
    #     else:
    #         start_id = random.randint(0, num_frame_data - self.seq_len)
    #     frames = frame_data[start_id: start_id + self.seq_len: self.cfg.DATA.SAMPLING_RATE]
    #
    #     # 后面的可能不对
    #     frames = np.array(frames)
    #     #frames = frames.astype(double)
    #     frames = torch.tensor(frames)
    #     frames = frames / 255.0
    #     frames = frames - torch.tensor(self.cfg.DATA.MEAN)
    #     frames = frames / torch.tensor(self.cfg.DATA.STD)
    #     # T H W C -> C T H W.
    #     frames = frames.permute(3, 0, 1, 2)
    #
    #     label = self._labels[index]
    #     # labels_mt = 1 if label == 34 else 0
    #     frames = utils.pack_pathway_output(self.cfg, frames)
    #
    #     if index not in self.index_list:
    #         self.index_list.append(index)
    #
    #     return frames, label, index, {}
    # def __getitem__(self, index):
    #     if self.mode in ["train", "val"]:
    #         # -1 indicates random sampling.
    #         temporal_sample_index = -1
    #         spatial_sample_index = -1
    #         min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
    #         max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
    #         crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
    #     elif self.mode in ["test"]:
    #         temporal_sample_index = (
    #             self._spatial_temporal_idx[index]
    #             // self.cfg.TEST.NUM_SPATIAL_CROPS)
    #         spatial_sample_index = (
    #             self._spatial_temporal_idx[index]
    #             % self.cfg.TEST.NUM_SPATIAL_CROPS)
    #         min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
    #         assert len({min_scale, max_scale, crop_size}) == 1
    #     else:
    #         raise NotImplementedError(
    #             "Does not support {} mode".format(self.mode)
    #         )
    #
    #     for _ in range(self._num_retries):     # 尝试多次
    #         frame_data = None
    #         video_container = None
    #         try:
    #             # print(self._path_to_videos[index], '*****')
    #             frame_data, size = get_frames(self._path_to_videos[index])
    #             # print(len(frame_data), '0000000')
    #         except Exception as e:
    #             print("Failed to load video from {} with error {}".format(
    #                     self._path_to_videos[index], e))
    #
    #         num_frame_data = len(frame_data)
    #         if num_frame_data < self.seq_len:
    #             frame_data, indices = process_short_video(frame_data, self.seq_len)
    #             start_id = 0
    #         else:
    #             start_id = random.randint(0, num_frame_data-self.seq_len)
    #         frames = frame_data[start_id: start_id + self.seq_len: self.cfg.DATA.SAMPLING_RATE]
    #
    #         # ind = torch.linspace(0, len(frames), self.cfg.DATA.NUM_FRAMES).long()
    #         # frames = frames[ind]
    #
    #         # 后面的可能不对
    #         frames = np.array(frames)
    #         #frames = frames.astype(double)
    #         frames = torch.tensor(frames)
    #         frames = frames / 255.0
    #         frames = frames - torch.tensor(self.cfg.DATA.MEAN)
    #         frames = frames / torch.tensor(self.cfg.DATA.STD)
    #         # T H W C -> C T H W.
    #         frames = frames.permute(3, 0, 1, 2)
    #         # print('frames.shape: {}'.format(frames.shape))
    #         #
    #         # frames = self.spatial_sampling(
    #         #     frames,
    #         #     spatial_idx=spatial_sample_index,
    #         #     min_scale=min_scale,
    #         #     max_scale=max_scale,
    #         #     crop_size=crop_size,
    #         # )
    #         # print(frames.shape)
    #         label = self._labels[index]
    #         frames = utils.pack_pathway_output(self.cfg, frames)
    #         # frames = preprocess_action_data(frames, self.cfg)
    #
    #         return frames, label, index, {}


    def __len__(self):
        return len(self._path_to_videos)

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames


def preprocess_action_data(frames, cfg):
    """
    数据预处理，归一化，减均值等，分别为slow和fast准备数据
    主要包括归一化，通道变换，从序列中取帧
    :param frames:
    :param cfg:
    :return:
    """
    inputs = torch.as_tensor(frames)#.float()
    inputs = inputs / 255.0
    # Perform color normalization.
    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
    inputs = inputs / torch.tensor(cfg.DATA.STD)
    # T H W C -> C T H W.
    inputs = inputs.permute(3, 0, 1, 2)

    # 1 C T H W.
    inputs = inputs.unsqueeze(0)

    # Sample frames for the fast pathway.
    index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
    fast_pathway = torch.index_select(inputs, 2, index)       # 取出32帧，用于fastway

    # Sample frames for the slow pathway.
    index = torch.linspace(0, fast_pathway.shape[2] - 1,
                           fast_pathway.shape[2] // cfg.SLOWFAST.ALPHA).long()
    slow_pathway = torch.index_select(fast_pathway, 2, index)   # 取出8帧，用于slowway
    inputs = [slow_pathway, fast_pathway]

    # Transfer the data to the current GPU device.
    if isinstance(inputs, (list,)):
        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda(non_blocking=True)
    else:
        inputs = inputs.cuda(non_blocking=True)
    return inputs



# class Kinetics(torch.utils.data.Dataset):
#     def __init__(self, cfg, mode, num_retries=10):
#         assert mode in ["train", "val", "test",], "Split '{}' not supported for Kinetics".format(mode)
#         self.mode = mode
#         self.cfg = cfg
#
#         self._video_meta = {}
#         self._num_retries = num_retries     # 这个参数是干什么的
#
#         if self.mode in ["train", "val"]:
#             self._num_clips = 1     # 这个参数是干什么的
#         elif self.mode in ["test"]:
#             self._num_clips = (
#                 cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
#             )
#
#         self._construct_loader()
#
#     def _construct_loader(self):
#         # 创建视频加载器
#         path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode))      # 获取对应的csv文件的路径
#         assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)
#
#         self._path_to_videos = []          # 用来存储视频的路径
#         self._labels = []                  # 用来存储视频的标签
#         self._spatial_temporal_idx = []    # 这个不知道， 用来空间抽样
#
#         with open(path_to_file, 'r') as f:
#             reader = csv.reader(f)
#             for clip_idx, (path, row) in enumerate(reader):
#                 for idx in range(self._num_clips):
#                     self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, path))      # 这里是得到了所有视频的路径
#                     self._labels.append(int(label))                                                 # 得到所有的label
#                     self._spatial_temporal_idx.append(idx)                                          # 空间抽样的索引，表示第几段视频，其实没用上
#                     self._video_meta[clip_idx * self._num_clips + idx] = {}
#
#     def __getitem__(self, index):
#         if self.mode in ["train", "val"]:
#             # -1 indicates random sampling.
#             temporal_sample_index = -1
#             spatial_sample_index = -1
#             min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
#             max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
#             crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
#         elif self.mode in ["test"]:
#             temporal_sample_index = (
#                 self._spatial_temporal_idx[index]
#                 // self.cfg.TEST.NUM_SPATIAL_CROPS)
#             spatial_sample_index = (
#                 self._spatial_temporal_idx[index]
#                 % self.cfg.TEST.NUM_SPATIAL_CROPS)
#             min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
#             assert len({min_scale, max_scale, crop_size}) == 1
#         else:
#             raise NotImplementedError(
#                 "Does not support {} mode".format(self.mode)
#             )
#
#         for _ in self._num_retries:     # 尝试多次
#             frame_data = None
#             video_container = None
#             try:
#                 frame_data, size = get_frames(self._path_to_videos[index])
#                 video_container = cv2.VideoCapture(self._path_to_videos[index])
#             except Exception as e:
#                 print("Failed to load video from {} with error {}".format(
#                         self._path_to_videos[index], e))
#
#             if frame_data is None:
#                 index = random.randint(0, len(self._path_to_videos) - 1)
#
#             frames = tianchi_decode(
#                 video_container,
#                 self.cfg.DATA.SAMPLING_RATE,
#                 self.cfg.DATA.NUM_FRAMES,
#                 temporal_sample_index,
#                 self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
#                 video_meta=self._video_meta[index],
#                 target_fps=30,
#             )
































