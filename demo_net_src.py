import os, sys
from time import time

import numpy as np
import pandas as pd
import cv2
import torch
import math

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
from slowfast.utils import logging
from slowfast.utils import misc
from slowfast.datasets import cv2_transform
from slowfast.models import model_builder
from slowfast.datasets.cv2_transform import scale

from functools import reduce

logger = logging.get_logger(__name__)
np.random.seed(20)


def video_data(cfg,videoname):
    cap = cv2.VideoCapture(videoname)
    frames = []
    k = 0
    while True:
       ret,frame = cap.read()
       if ret:
           frames.append(frame)
       else:
           break

    seq_len = cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE
    if len(frames) >= seq_len:
       return frames
    else:
       num = seq_len - len(frames) + 1
       indices = np.linspace(start=0, stop=len(frames)-1, num=num, dtype='int32')
       ind = np.arange(len(frames)).astype('int32')
       indices = list(np.sort(np.append(indices, ind)))     
       frames_dst = []
       for i in range(len(indices)):
           frames_dst.append(frames[indices[i]])
       return frames_dst


def update_result(label_ids, preds, label_list):
    predict_label_list = []
    predict_score_list = []
    predict_label_dict = dict()    

    pred_labels = list()
    [pred_labels.append(label_list[x]) for x in label_ids]
    scores = []
    [scores.append(preds.squeeze()[x].cpu().detach().numpy()) for x in label_ids]
    for l, s in zip(pred_labels, scores):
         if l in predict_label_dict.keys():
              predict_label_dict[l].append(float(s))
         else:
              predict_label_dict[l] = [float(s)]
    [predict_label_list.append(x) for x in pred_labels]
    [predict_score_list.append(float(x)) for x in scores]         

    if len(predict_label_dict) > 0:
            num_list = list()
            [num_list.append(len(x)) for x in predict_label_dict.values()]
            thresh = math.ceil(sum(num_list) / len(predict_label_dict.keys()))  # 可以改成向下取整 ****

            last_label, last_score = list(), list()
            if thresh < max(num_list):
                # 此时说明存在大于该阈值的标签
                threshed_list = []
                [threshed_list.append(len(v)) for k, v in predict_label_dict.items() if len(v) > thresh]
                threshed_list.sort(reverse=True)

                for x in threshed_list:
                    tmp_label = []
                    tmp_score = []
                    for k, v in predict_label_dict.items():
                        if len(v) == x:
                            tmp_label.append(k)
                            tmp_score.append(max(v))  # 可以改成取均值 ***
                    if len(tmp_label) > 0:
                        indices = list(np.argsort(np.array(tmp_score)))
                        [last_label.append(tmp_label[tmp]) for tmp in indices]
                        [last_score.append(tmp_score[tmp]) for tmp in indices]
            else:
                if max(num_list) == min(num_list):  # max(num_list) - min(num_list) ==1
                    # 此时，标签出现的次数相等，取出所有的标签
                    tmp_label = []
                    tmp_score = []
                    [tmp_label.append(k) for k in predict_label_dict.keys()]
                    [tmp_score.append(max(v)) for v in predict_label_dict.values()]

                    indices = list(np.argsort(np.array(tmp_score)))
                    [last_label.append(tmp_label[tmp]) for tmp in indices]
                    [last_score.append(tmp_score[tmp]) for tmp in indices]
                else:
                    # 此时标签出现的次数不一定相等，如9， 8，8，thresh=9
                    n = max(num_list)
                    [last_label.append(k) for k, v in predict_label_dict.items() if len(v) == n]
                    [last_score.append(max(predict_label_dict[x])) for x in last_label]
    else:
        last_label, last_score = list(), list()
    func = lambda x, y: x if y in x else x + [y]
    last_label = reduce(func, [[], ] + last_label)
    last_score = reduce(func, [[], ] + last_score)
    return last_label, last_score


def kinetics_predict(cfg,model,frames_data,labels):
    frames = []
    for frame in frames_data:
        frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_processed = scale(cfg.DATA.TEST_CROP_SIZE, frame_processed)
        frames.append(frame_processed)
    inputs = torch.as_tensor(np.array(frames)).float()
    inputs = inputs / 255.0
    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
    inputs = inputs / torch.tensor(cfg.DATA.STD)
    inputs = inputs.permute(3, 0, 1, 2)
    inputs = inputs.unsqueeze(0)
    # Sample frames for the fast pathway.
    index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
    fast_pathway = torch.index_select(inputs, 2, index)
    # Sample frames for the slow pathway.
    index = torch.linspace(0, fast_pathway.shape[2] - 1,
                                    fast_pathway.shape[2]//cfg.SLOWFAST.ALPHA).long()
    slow_pathway = torch.index_select(fast_pathway, 2, index)
    inputs = [slow_pathway, fast_pathway]
    # Transfer the data to the current GPU device.
    if isinstance(inputs, (list,)):
        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda(non_blocking=True)
    else:
        inputs = inputs.cuda(non_blocking=True)
    preds = model(inputs)
    # Gather all the predictions across all the devices to perform ensemble.
    if cfg.NUM_GPUS > 1:
        preds = du.all_gather(preds)[0]
    label_ids = torch.nonzero(preds.squeeze() > .1).reshape(-1).cpu().detach().numpy()
    pred_labels = labels[label_ids]
    print(pred_labels)
    last_label,last_score = update_result(label_ids, preds, labels)
    print(last_label,last_score)
    return last_label,last_score,pred_labels


def demo(cfg):
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    model.eval()
    misc.log_model_info(model)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        ckpt = cfg.TEST.CHECKPOINT_FILE_PATH
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        ckpt = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        ckpt = cfg.TRAIN.CHECKPOINT_FILE_PATH
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    cu.load_checkpoint(
        ckpt,
        model,
        cfg.NUM_GPUS > 1,
        None,
        inflation=False,
        convert_from_caffe2= "caffe2" in [cfg.TEST.CHECKPOINT_TYPE, cfg.TRAIN.CHECKPOINT_TYPE],
    )

    # Load the labels of Kinectics-400 dataset
    labels_df = pd.read_csv(cfg.DEMO.LABEL_FILE_PATH)
    labels = labels_df['name'].values

    """
    # predict video_list
    writer = open("huawei_res.txt","w")
    reader = open("huawei.txt","r")
    for line in reader:
        videoname = line.strip()
        frames_data = video_data(cfg,videoname)
        last_label,last_score,pred_labels = kinetics_predict(cfg,model,frames_data,labels)   
        res = videoname
        for i in range(len(pred_labels)):
            res = res + "," + pred_labels[i]
        res = res + "_________"
        for i in  range(len(last_label)):
            res = res + "," + last_label[i] + "," + str(last_score[i])
        writer.writelines(res+"\n")
    """
    
    # predict single video
    videoname = cfg.DEMO.DATA_SOURCE
    frames_data = video_data(cfg,videoname)
    last_label,last_score,pred_labels = kinetics_predict(cfg,model,frames_data,labels)    
    

    """
    # predict
    frames = []
    for frame in frames_data:
        frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_processed = scale(cfg.DATA.TEST_CROP_SIZE, frame_processed)
        frames.append(frame_processed)
    inputs = torch.as_tensor(np.array(frames)).float()
    inputs = inputs / 255.0
    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
    inputs = inputs / torch.tensor(cfg.DATA.STD)
    inputs = inputs.permute(3, 0, 1, 2)
    inputs = inputs.unsqueeze(0)
    # Sample frames for the fast pathway.
    index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
    fast_pathway = torch.index_select(inputs, 2, index)
    # Sample frames for the slow pathway.
    index = torch.linspace(0, fast_pathway.shape[2] - 1,
                                    fast_pathway.shape[2]//cfg.SLOWFAST.ALPHA).long()
    slow_pathway = torch.index_select(fast_pathway, 2, index)
    inputs = [slow_pathway, fast_pathway]
    # Transfer the data to the current GPU device.
    if isinstance(inputs, (list,)):
        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda(non_blocking=True)
    else:
        inputs = inputs.cuda(non_blocking=True)
    preds = model(inputs)
    # Gather all the predictions across all the devices to perform ensemble.
    if cfg.NUM_GPUS > 1:
        preds = du.all_gather(preds)[0]
    label_ids = torch.nonzero(preds.squeeze() > .1).reshape(-1).cpu().detach().numpy()
    pred_labels = labels[label_ids]
    print(pred_labels)
    last_label,last_score = update_result(label_ids, preds, labels)
    print(last_label,last_score)
    """
