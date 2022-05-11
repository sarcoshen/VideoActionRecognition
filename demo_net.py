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

import argparse
import sys
import os

from slowfast.config.defaults import get_cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

logger = logging.get_logger(__name__)
np.random.seed(20)


front_face = cv2.CascadeClassifier(r'/data1/anaconda3/envs/scene_action/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(r'/data1/anaconda3/envs/scene_action/lib/python3.6/site-packages/cv2/data/haarcascade_profileface.xml')
left_eye = cv2.CascadeClassifier(r'/data1/anaconda3/envs/scene_action/lib/python3.6/site-packages/cv2/data/haarcascade_lefteye_2splits.xml')
#right_eye = cv2.CascadeClassifier(r'/data1/anaconda3/envs/scene_action/lib/python3.6/site-packages/cv2/data/haarcascade_righteye_2splits.xml')
lower_body = cv2.CascadeClassifier(r'/data1/anaconda3/envs/scene_action/lib/python3.6/site-packages/cv2/data/haarcascade_lowerbody.xml')


def is_exist_human(frames):
    frames_len = len(frames)
    start_len = int(frames_len / 4)
    end_len = int(frames_len / 4 * 3)
    new_frames = []
    for i in range(start_len,end_len):
         new_frames.append(frames[i])
    if end_len - start_len + 1 > 30:
         index = torch.linspace(0, len(new_frames)-1, 30)
         #print("index: ",index)
         index = torch.clamp(index, 0, len(new_frames) - 1).long()
         print("index: ",index)
         tmp_frames = []
         for k in index:
             tmp_frames.append(new_frames[k])
         new_frames = tmp_frames
    print("frames: ",len(new_frames))      
    human_num = 0
    for i in range(len(new_frames)):
        gray = cv2.cvtColor(new_frames[i],cv2.COLOR_BGR2GRAY)
        f_faces = front_face.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(10,10),flags=cv2.CASCADE_SCALE_IMAGE)
        p_faces = profile_face.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(10,10),flags=cv2.CASCADE_SCALE_IMAGE)
        l_eyes = left_eye.detectMultiScale(gray)
        #r_eyes = right_eye.detectMultiScale(gray)
        l_bodies = lower_body.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(20,20),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(f_faces)>0 or len(p_faces)>0 or len(l_eyes)>0 or len(l_bodies)>0:
              human_num += 1
    if human_num > 5:
        return True
    else:
        return False
          


def random_short_side_scale_jitter(images, min_size, max_size):
    size = int(round(np.random.uniform(min_size, max_size)))
    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    return (
        torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
    )

def random_crop(images, size):
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]
    return cropped


def image_scale(size, image):
    height = image.shape[0]
    width = image.shape[1]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return image
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    img = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training, testing, and demo pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        #default = "./SLOWFAST_4x16_R50.yaml",
        #default = "./I3D_NLN_8x8_R50.yaml",
        default="./SLOWFAST_8x8_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()    



def load_config(args):
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg



def video_data(cfg,videoname):
    cap = cv2.VideoCapture(videoname)
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_frames = []
    k = 0
    while True:
       ret,frame = cap.read()
       if ret:
           src_frames.append(frame)
       else:
           break

    src_frames = [image_scale(cfg.DATA.TEST_CROP_SIZE,frame) for frame in src_frames]
    frames = []
    for i in range(len(src_frames)):
       if i > 2 and i < len(src_frames) - 20:
           frames.append(src_frames[i])
    if len(frames) < 30:
       frames = src_frames

    seq_len = cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE
    if len(frames) >= seq_len:
       return frames,fps
    else:
       num = seq_len - len(frames) + 1
       indices = np.linspace(start=0, stop=len(frames)-1, num=num, dtype='int32')
       ind = np.arange(len(frames)).astype('int32')
       indices = list(np.sort(np.append(indices, ind)))     
       frames_dst = []
       for i in range(len(indices)):
           frames_dst.append(frames[indices[i]])
       return frames_dst,fps


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



def one_predict(cfg,model,frames,labels):
    inputs = torch.as_tensor(np.array(frames)).float()
    inputs = inputs / 255.0
    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
    inputs = inputs / torch.tensor(cfg.DATA.STD)
    inputs = inputs.permute(3, 0, 1, 2)

    """
    # crop
    min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
    max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
    crop_size = cfg.DATA.TRAIN_CROP_SIZE
    inputs = random_short_side_scale_jitter(inputs, min_scale, max_scale)
    inputs = random_crop(inputs, crop_size)
    """

    inputs = inputs.unsqueeze(0)
    # Sample frames for the fast pathway.
    index = torch.linspace(10, inputs.shape[2] - 20, cfg.DATA.NUM_FRAMES).long()
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
    label_ids = torch.nonzero(preds.squeeze() > 0.01).reshape(-1).cpu().detach().numpy()
    pred_labels = labels[label_ids]
    predict_list = []
    for x in label_ids:
        predict_list.append([labels[x],float(preds.squeeze()[x].cpu().detach().numpy())])
    predict_list.sort(key=lambda x:x[1],reverse=True)
    return predict_list


def kinetics_predict(cfg,model,frames_data,fps,labels):
    frames = []
    for frame in frames_data:
        frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_processed = scale(cfg.DATA.TEST_CROP_SIZE, frame_processed)
        frames.append(frame_processed)
    calc_frames = int(7 * fps)
    iter_frames = int(4 * fps)
    calc_nums = int(len(frames) / 11 / fps) - 1
    final_res = 'None'
    final_score = 0

    # predict
    if calc_nums >= 0:
        label_score_list = []
        for i in range(calc_nums):
            sub_frames = []
            for j in range(i*iter_frames,i*iter_frames+calc_frames):
                if j >= len(frames):
                   break
                sub_frames.append(frames[j])
            if len(sub_frames) < 50 or len(sub_frames) < calc_frames:
                continue
            predict_list = one_predict(cfg,model,sub_frames,labels)
            max_score = predict_list[0][1]
            max_label = predict_list[0][0]
            label_score_list.append([max_label,max_score])

        if len(label_score_list) < 1:
            return 'None',0

        max_score = label_score_list[0][1]
        max_label = label_score_list[0][0]
        for i in range(len(label_score_list)):
            if label_score_list[i][1] > max_score:
                max_score = label_score_list[i][1]
                max_label = label_score_list[i][0]
        max_label_nums = 0
        for i in range(len(label_score_list)):
            if label_score_list[i][0] == max_label:
                max_label_nums += 1
        if max_label_nums >= 2:
            return max_label,max_score
        else:
            return 'None',0            
    else:
        predict_list = one_predict(cfg,model,frames,labels)
        final_label = predict_list[0][0]
        final_score = predict_list[0][1]
        return final_label,final_score
    
    """
    if calc_nums >= 2:
        max_score = 0
        res_dict = {}
        for tmp_label in labels:
            res_dict[tmp_label] = [0,0] # number,score
        res = 'None'
        for i in range(calc_nums):
            sub_frames = []
            for j in range(i*iter_frames,i*iter_frames+calc_frames):
                if j < len(frames) - 1:
                   sub_frames.append(frames[j])
            if len(sub_frames) < 100:
                continue
            predict_list = one_predict(cfg,model,sub_frames,labels)
            #print(predict_list)
            if predict_list[0][1] > max_score:
                max_score = predict_list[0][1]
                res = predict_list[0][0]
            for k in range(len(predict_list)):
                tmp_label = predict_list[k][0]
                tmp_score = predict_list[k][1]
                res_dict[tmp_label] = [res_dict[tmp_label][0]+1,res_dict[tmp_label][1]+tmp_score]
        if max_score > 0.5:
            final_res = res
            final_score = max_score
        else:
            max_score = 0
            res = 'None'
            for tmp_label in res_dict:
                if res_dict[tmp_label][1] > max_score:
                      max_score = res_dict[tmp_label][1]
                      res = tmp_label
            final_res = res
            final_score = max_score
    else:
        predict_list = one_predict(cfg,model,frames,labels)
        final_res = predict_list[0][0]
        final_score = predict_list[0][1]
    
    return final_res, final_score
    """
    
    #predict_list = one_predict(cfg,model,frames,labels)
    #final_res = predict_list[0][0]
    #final_score = predict_list[0][1]
    #return final_res,final_score    
    


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
        print(ckpt)
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
    print(labels) 
    print(ckpt)
    
   
    fr = open('/data2/shenxiaolei02/test_data/movie_test.txt',"r")
    fw1 = open('movie_17_455.txt',"w")
    for line in fr:
        filepath = line.strip().split(",")[0].strip()
        label = line.strip().split(",")[1].strip()
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            frames_data,fps = video_data(cfg,filepath)
            #if False == is_exist_human(frames_data):
            #   continue
            #print(filepath)
            #fw1.writelines(line.strip()+"\n")
            
            
            res_label,res_score = kinetics_predict(cfg,model,frames_data,fps,labels)
            #if res_score < 0.4:
            #   fw1.writelines(filepath + "," + str(res_label) + "," + str(res_score) + "\n")
            
            #if res_score >= 0.0 or res_score < 0.3:
            fw1.writelines(filepath + "," + label + "," + str(res_label) + "," + str(res_score) + "\n")
            print(filepath,label,res_label,res_score)
            
         
     
    """ 
    # predict video_list
    label_dict = {}
    idx_dict = {}
    f = open('all_labels_16.csv',"r")
    idx = 0
    label_dict['None'] = 16
    idx_dict['16'] = 'None'
    for row in f:
        line = row.strip()
        if idx > 0:
            label_dict[line.split(",")[1]] = int(line.split(",")[0])
            idx_dict[line.split(",")[0]] = line.split(",")[1]
        idx += 1
    print(label_dict)
    print(idx_dict)
    result_dict = {}
   
    for i in range(17):
        result_dict[i] = [0,0]
    reader = open("../../sf_data/test.csv","r")
    writer = open("test_16_329.txt","w")
    for row in reader:
        videoname = row.strip().split(",")[0]
        label = row.strip().split(",")[-1]
        frames_data,fps = video_data(cfg,videoname)
        res_label,res_score = kinetics_predict(cfg,model,frames_data,fps,labels)  
        
        result_dict[int(label_dict[label])][0] += 1 
        res = videoname + "," + label + "," + res_label + "," + str(res_score)
        writer.writelines(res+"\n")
        if res_label == label:
            result_dict[int(label_dict[label])][1] += 1
    print(result_dict)
    for label in result_dict:
        num1 = str(result_dict[int(label_dict[label])][0])
        num2 = str(result_dict[int(label_dict[label])][1])
        writer.writelines(label+","+num1+","+num2+"\n")
    """
   

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    demo(cfg)   




