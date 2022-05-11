import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import cv2




count_dict = {}
count_dict['开车'] = 0
count_dict['牵手'] = 0
count_dict['打斗'] = 0
count_dict['拥抱'] = 0
count_dict['亲吻'] = 0
count_dict['持刀'] = 0
count_dict['打篮球'] = 0
count_dict['打电话'] = 0
count_dict['吸烟'] = 0
count_dict['吃东西'] = 0
count_dict['持枪'] = 0
count_dict['芭蕾舞'] = 0
count_dict['群体跳舞'] = 0
count_dict['探戈舞'] = 0
count_dict['唱歌'] = 0
count_dict['挥手'] = 0
count_dict['哭泣'] = 0
count_dict['弹吉他'] = 0
count_dict['None'] = 0
count_dict['摸头'] = 0
count_dict['摸脸'] = 0
count_dict['西式婚礼'] = 0
count_dict['回头'] = 0
count_dict['吻额头']  = 0
count_dict['脱衣'] = 0
count_dict['骑马'] = 0
count_dict['剃头'] = 0
count_dict['游泳'] = 0

label_dict = {}
label_dict['driving'] = '开车'
label_dict['qian_shou'] = '牵手'
label_dict['scuffling'] = '打斗'
label_dict['hugging'] = '拥抱'
label_dict['kissing'] = '亲吻'
label_dict['chi_dao'] = '持刀'
label_dict['playing_basketball'] = '打篮球'
label_dict['da_dianhua'] = '打电话'
label_dict['smoking'] = '吸烟'
label_dict['eating'] = '吃东西'
label_dict['chi_qiang'] = '持枪'
label_dict['dancing_ballet'] = '芭蕾舞'
label_dict['dancing_qunti'] = '群体跳舞'
label_dict['tango_dancing'] = '探戈舞'
label_dict['singing'] = '唱歌'
label_dict['hui_shou'] = '挥手'
label_dict['crying'] = '哭泣'
label_dict['playing_guitar'] = '弹吉他'
label_dict['riding_horse'] = '骑马'
label_dict['swimming'] = '游泳'
label_dict['shaving_head'] = '剃头'

dancing = []
dancing.append('芭蕾舞')
dancing.append('群体跳舞')
dancing.append('探戈舞')

"""
label_files = []
none_files = []
score_thd = 0.95
fr = open('movie_493_res.txt',"r")
fw = open('none_0p95_493.txt',"w")
for line in fr:
    filepath = line.strip().split(",")[0]
    gt_label = line.strip().split(",")[1].split(":")
    gt_label_0 = line.strip().split(",")[1]
    predict_label = line.strip().split(",")[2]
    if line.strip().split(",")[1] == '哭泣':
        continue
    if line.strip().split(",")[1] == '摸头':
        continue
    if line.strip().split(",")[1] == '摸脸':
        continue
    if line.strip().split(",")[1] == '西式婚礼':
        continue
    if line.strip().split(",")[1] == '回头':
        continue
    if line.strip().split(",")[1] == '吻额头':
        continue
    if line.strip().split(",")[1] == '脱衣':
        continue
    #if line.strip().split(",")[1] == '持刀':
    #    continue

    
    if predict_label == 'playing_guitar':
        continue
    #if predict_label == 'chi_dao':
    #    continue
    #if predict_label == 'crying':
    #    continue
    if predict_label == 'swimming':
        continue
    if predict_label == 'riding_horse':
        continue
    if predict_label == 'shaving_head':
        continue
    
    if line.strip().split(",")[1] == 'None':
        none_files.append(line.strip())
        continue
    label_files.append(line.strip())

print(len(label_files))
print(len(none_files))

k1 = 0
k2 = 0
k3 = 0
for line in label_files:
    gt_label = line.split(",")[1].split(":")
    predict_label = line.split(",")[2]
    score = line.split(",")[-1]
    if float(score) >= score_thd:
        if label_dict[predict_label] in gt_label or label_dict[predict_label] in dancing:
            k1 += 1
        k2 += 1
    k3 += 1
k4 = 0
k5 = 0

for line in none_files:
    gt_label = line.split(",")[1]
    predict_label = line.split(",")[2]
    score = line.split(",")[-1]
    if float(score) >= score_thd:
        k4 += 1
        fw.writelines(line+"\n")
    k5 += 1
    
print('labels: ',k1,k2,k3)
print('none: ',k4,k5)
"""



score_thd = 0.95
fr = open('movie_test_497.txt',"r")
fr2 = open('movie_test_455.txt',"r")
k1 = 0
k2 = 0
k3 = 0
k4 = 0
file_list = []
for line in fr:
    filepath = line.strip().split(",")[0]
    gt_label = line.strip().split(",")[1].split(":")
    if len(gt_label) == 1:
        count_dict[gt_label[0]] += 1
    if len(gt_label) > 1:
        for tmp_label in gt_label:
            count_dict[tmp_label] += 1
    predict_label = line.strip().split(",")[2]
    
    if line.strip().split(",")[1] == '哭泣':
        continue
    if line.strip().split(",")[1] == '持刀':
        continue
       
    if line.strip().split(",")[1] == '摸头':
        continue
    if line.strip().split(",")[1] == '摸脸':
        continue
    if line.strip().split(",")[1] == '西式婚礼':
        continue
    if line.strip().split(",")[1] == '回头':
        continue
    if line.strip().split(",")[1] == '吻额头':
        continue    
    if line.strip().split(",")[1] == '脱衣':
        continue
    
    if predict_label == 'playing_guitar':
        continue
    if predict_label == 'chi_dao':
        continue
    if predict_label == 'crying':
        continue
    if predict_label == 'swimming':
        continue
    if predict_label == 'riding_horse':
        continue
    if predict_label == 'shaving_head':
        continue

    file_list.append(filepath)
    score = line.strip().split(",")[-1]

    if line.strip().split(",")[1]  != 'None':
        k3 += 1
    if float(score) >= score_thd:
        if label_dict[predict_label] in gt_label or label_dict[predict_label] in dancing: 
            k1 += 1
        k2 += 1
    k4 += 1    
print(score_thd,k1,k2,k3,k4)
k1 = 0
k2 = 0
k4 = 0
fw2 = open('c_test.txt',"w")
for line in fr2:
    filepath = line.strip().split(",")[0]
    gt_label = line.strip().split(",")[1].split(":")
    predict_label = line.strip().split(",")[2]
    score = line.strip().split(",")[-1]
    if filepath in file_list:
       fw2.writelines(filepath+","+line.strip().split(",")[1]+"\n")
       if float(score) >= score_thd:
            if label_dict[predict_label] in gt_label or label_dict[predict_label] in dancing:
                k1 += 1
            k2 += 1
       k4 += 1
print(score_thd,k1,k2,k4)






"""
label_dict = {}
label_dict['qian_shou'] = 0
label_dict['kissing'] = 0
label_dict['scuffling'] = 0
label_dict['driving'] = 0
label_dict['eating'] = 0
label_dict['hugging'] = 0
label_dict['chi_qiang'] = 0
label_dict['hui_shou'] = 0
label_dict['dancing_ballet'] = 0
label_dict['tango_dancing'] = 0
label_dict['chi_dao'] = 0
label_dict['singing'] = 0
label_dict['playing_basketball'] = 0
label_dict['smoking'] = 0
label_dict['da_dianhua'] = 0
label_dict['dancing_qunti'] = 0
label_dict['playing_guitar'] = 0
label_dict['riding_horse'] = 0
label_dict['swimming'] = 0
label_dict['shaving_head'] = 0
label_dict['crying'] = 0


fr = open('none_0p9_388.txt',"r")
for line in fr:
    predict_label = line.strip().split(",")[2]
    gt_label = line.strip().split(",")[1].strip().split(":")
    label_dict[predict_label] += 1
for label in label_dict:
    print(label,label_dict[label])
"""



