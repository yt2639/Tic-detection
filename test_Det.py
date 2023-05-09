import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from create_video_table import get_data_table_DET0102
from ssd1d import *
from data_preparation import * 
from cal_map import *
from anchors import *
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

args = sys.argv

gpu_device = args[2]
device = torch.device('cuda:{}'.format(gpu_device)) if torch.cuda.is_available() else torch.device('cpu')
print('gpu device:',device)

usecols = []
usecols.extend(['frame',' timestamp',' confidence',' success',' gaze_angle_x',' gaze_angle_y',
                ' pose_Tx',' pose_Ty',' pose_Tz',' pose_Rx',' pose_Ry',' pose_Rz'])
AU_i = [' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',' AU17_r',' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r']
AU_p = [' AU01_c',' AU02_c',' AU04_c',' AU05_c',' AU06_c',' AU07_c',' AU09_c',' AU10_c',' AU12_c',' AU14_c',' AU15_c',' AU17_c',' AU20_c',' AU23_c',' AU25_c',' AU26_c',' AU28_c',' AU45_c']
usecols.extend(AU_i)
usecols.extend(AU_p)


v1_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V1_HI.csv',
                                label_path='./Annots/DET0102_V1_HI.csv', 
                                usecols=usecols)
v1_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V1_LO.csv',
                                label_path='./Annots/DET0102_V1_LO.csv', 
                                usecols=usecols)
v1_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V1_NO.csv',
                                label_path='./Annots/DET0102_V1_NO.csv', 
                                usecols=usecols)


v2_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V2_HI.csv',
                                label_path='./Annots/DET0102_V2_HI.csv', 
                                usecols=usecols)
v2_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V2_LO.csv',
                                label_path='./Annots/DET0102_V2_LO.csv', 
                                usecols=usecols)
v2_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V2_NO.csv',
                                label_path='./Annots/DET0102_V2_NO.csv', 
                                usecols=usecols)


v3_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V3_HI.csv',
                                label_path='./Annots/DET0102_V3_HI.csv', 
                                usecols=usecols)
v3_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V3_LO.csv',
                                label_path='./Annots/DET0102_V3_LO.csv', 
                                usecols=usecols)
v3_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V3_NO.csv',
                                label_path='./Annots/DET0102_V3_NO.csv', 
                                usecols=usecols)


v4_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V4_HI.csv',
                                label_path='./Annots/DET0102_V4_HI.csv', 
                                usecols=usecols)
v4_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V4_LO.csv',
                                label_path='./Annots/DET0102_V4_LO.csv', 
                                usecols=usecols)
v4_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V4_NO.csv',
                                label_path='./Annots/DET0102_V4_NO.csv', 
                                usecols=usecols)


test_name = args[1]
all_video_names = ['v1_hi', 'v1_lo', 'v1_no',
                   'v2_hi', 'v2_lo', 'v2_no',
                   'v3_hi', 'v3_lo', 'v3_no',
                   'v4_hi', 'v4_lo', 'v4_no']
train_video_names = [n for n in all_video_names if test_name not in n]
test_video_names = [n for n in all_video_names if test_name in n]
print(train_video_names)
print(test_video_names)
CAL_MAP_TABLE = {n:eval(n) for n in test_video_names}

CLASSES = [0,1,2,3,4,5,6,7,8,9,10,11,12]
FEATURES = AU_i
# sliding window generate samples
all_train_gt_boxes = []
for data_name in train_video_names:
    data = eval(data_name)
    train_all_X = data[FEATURES].values
    label_matrix = data[CLASSES].values
    frames = data['frame'].values
    timestamp = data[' timestamp'].values
    
    _, _, train_gt_boxes, _ = get_feat_and_gt_boxes(
                                    train_all_X, label_matrix,
                                    training_stride=50, sample_length=416)
    all_train_gt_boxes.extend(train_gt_boxes)
    
all_train_gt_boxes, all_train_gt_boxes_names = modify_gt_boxes(all_train_gt_boxes, thresh=3)
all_train_gt_boxes = [torch.from_numpy(e).float() for e in all_train_gt_boxes]

###
all_test_X = []
all_test_general_labels = []
all_test_video_general_labels = {}
all_test_gt_boxes = []
all_test_gt_boxes_names = []
all_test_global_gt_boxes = {}
for data_name in test_video_names:
    data = eval(data_name)
    test_all_X = data[FEATURES].values.astype(np.float32)
    label_matrix = data[CLASSES].values
    frames = data['frame'].values
    timestamp = data[' timestamp'].values
    general_labels = data['general_labels'].values
    all_test_video_general_labels[data_name] = general_labels

    test_n_samples, test_X, test_general_labels, test_gt_boxes, _ = get_feat_and_gt_boxes_with_general_labels(
                                                                        test_all_X, label_matrix, general_labels,
                                                                        training_stride=50, sample_length=416)

    n_samples, test_gt_boxes_names, test_global_gt_boxes = get_global_gt_boxes_and_names(
                                                                data_name,label_matrix,frames,timestamp,'time',
                                                                training_stride=50, sample_length=416)
    
    all_test_X.append(test_X.astype(np.float32))
    all_test_general_labels.append(test_general_labels)
    all_test_gt_boxes.extend(test_gt_boxes)
    all_test_gt_boxes_names.extend(test_gt_boxes_names)
    all_test_global_gt_boxes[data_name] = torch.from_numpy(test_global_gt_boxes)
    
all_test_X = np.concatenate(all_test_X)
all_test_general_labels = np.concatenate(all_test_general_labels)

all_test_gt_boxes, all_test_gt_boxes_names = modify_gt_boxes(all_test_gt_boxes, all_test_gt_boxes_names, thresh=3)
all_test_gt_boxes = [torch.from_numpy(e).float() for e in all_test_gt_boxes]

all_test_X = torch.from_numpy(all_test_X).float()
all_test_general_labels = torch.from_numpy(all_test_general_labels).unsqueeze(1)

# max value normalize 
all_test_X = all_test_X / 5

print('==================')
print('read data done!')
print('==================')
print('Test_X:',all_test_X.shape)

BATCH_SIZE = int(args[4])
test_dataset = TicDataset_all_aug(all_test_X.permute(0,2,1), all_test_gt_boxes, mode='testing', 
                                    is_noise=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_function, drop_last=False, 
                            pin_memory=True)

print('Test batch_size = {}'.format(BATCH_SIZE))

## anchors
all_proposal_boxes = anchors_det0102()
# all_proposal_boxes = kmeans_anchors(all_train_gt_boxes)
all_proposal_boxes = all_proposal_boxes.to(device)

# training settings
is_focal = [False,True][0]
# net = Det1D(backbone_name='resnet18_large512', device=device,
#             thresh_low= -0.4, thresh_high=0.3, loss_type='DetLoss',
#             topk_match=8, iou_smooth=False, is_focal=is_focal, uni_mat=False, 
#             use_EIoU_loss=True, use_EIoU=True,
#             conf_threshold=0.2, nms_threshold=0.2, top_k=100)

model_file_directory = args[3]

import logging
# log_path = args[5]
log_path = 'test_results/test' + model_file_directory.split('/')[-1].lstrip('train') + '/test.log'


sapa = os.path.split(log_path)[0]
sapa += '/'
if not os.path.exists(sapa):
    os.makedirs(sapa)
print(sapa)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file = logging.FileHandler(log_path) # log_path
log_file.setLevel(logging.INFO)
logger.addHandler(log_file)
logger.info('log path: {}'.format(log_path))

# not_use_comet = True
writer = SummaryWriter(comment=log_path.split('/')[-1].strip('.log'))#, comet_config={"disabled": not_use_comet})

IOU_THRESH = 0.5

num_batch = len(test_dataset)//BATCH_SIZE
print('Testing data has {} samples, {}//{}={} batch'.format(len(test_dataset),len(test_dataset),BATCH_SIZE,num_batch))

metrics_dict = {}
for m in sorted(os.listdir(model_file_directory)):
    if '.pth' not in m:
        continue
    epoch = int(m.split('_')[-1].rstrip('.pth'))
    if epoch >= int(args[7]):

        pretrained_dict = torch.load('{}/{}'.format(model_file_directory,m), map_location='cpu')

        net = Det1D(backbone_name='resnet18_large512', device=device,
                    thresh_low= -0.4, thresh_high=0.3, loss_type='DetLoss',
                    topk_match=8, iou_smooth=False, is_focal=is_focal, uni_mat=False, 
                    use_EIoU_loss=True, use_EIoU=True,
                    conf_threshold=0.2, nms_threshold=0.2, top_k=100)
        net.load_state_dict(pretrained_dict)
        print('load model successful!')

        # single GPU:
        net.to(device)
        print('model moves to GPU done!')

        ##------------------------------------------------------------------ eval test

        net.eval()
        all_detections = []
        all_gt_boxes = []
        with torch.no_grad():
            for i,(cur_X,cur_gt_boxes) in enumerate(test_loader):
                cur_X = cur_X.to(device)
                cur_gt_boxes = [e.to(device) for e in cur_gt_boxes]
                detections = net(cur_X,all_proposal_boxes,None)
                all_detections.append(detections)
                all_gt_boxes.extend(cur_gt_boxes)

            all_detections = torch.cat(all_detections)    
            _,_,test_ap,test_auc, TP,MatchedGT,_,_, num_tp, num_fp, num_fn, _,_,_,_, _ = cal_tp_fp_fn(
                                                                                            test_video_names,
                                                                                            all_detections,all_test_gt_boxes_names,all_test_global_gt_boxes,
                                                                                            iou_threshold = IOU_THRESH, CAL_MAP_TABLE=CAL_MAP_TABLE)
        ##------------------------------------------------------------------ eval test
        logger.info('epoch:{} | AP:{:.4f} | TP:{} | FP:{} | FN:{}'.format(
                        epoch, test_ap, num_tp, num_fp, num_fn))
        
        writer.add_scalars('Test/metrics', {'AP': test_ap, 
                                            'TP#':num_tp, 'FP#':num_fp, 'FN#':num_fn}, epoch)

        metrics_dict[epoch] = [test_ap, num_tp, num_fp, num_fn]
        print('\n')

writer.close()

if args[6] == 'draw':
    summary_tab = pd.DataFrame(metrics_dict).T
    summary_tab.columns = ['AP','num_TP','num_FP','num_FN']
    summary_tab.sort_index(inplace=True)
    summary_tab.to_csv(sapa+'metrics.csv')

    x = np.arange(summary_tab.shape[0])
    plt.plot(x, summary_tab['AP'].values, label='AP')
    plt.xlabel('epoch',fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(sapa+'AP.png',bbox_inches=0,dpi=200)

    plt.cla()

    plt.plot(x, summary_tab['num_TP'].values, label='num_TP')
    plt.plot(x, summary_tab['num_FN'].values, label='num_FN')
    plt.xlabel('epoch',fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(sapa+'TP_FN.png',bbox_inches=0,dpi=200)

    plt.cla()
    
    plt.plot(x, summary_tab['num_FP'].values, label='num_FP')
    plt.xlabel('epoch',fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(sapa+'FP.png',bbox_inches=0,dpi=200)


### args = sys.argv
# args[0]: it is this file's name
# args[1]: test session, v1
# args[2]: share encoder, not_share
# args[3]: model directory, 'xx/xx/' which contains all the saved models
# args[4]: batch size
# args[5]: gpu no, 0,1,2,3
# args[6]: draw figure or not
# args[7]: starting epoch, test starting from this epoch

# python test_DetSeg.py v2 share saved_models/train_20211229_det0102_[Large512]_[LOSO_v2] 256 0 draw 0


