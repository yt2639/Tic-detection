import comet_ml
comet_ml.init(project_name='DET0102_CrossVal')

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import datetime
import time
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

v1_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V1_HI_Front/DET0102_V1_HI_Front.csv',
                                label_path='./Annots/DET0102_V1_HI_Front.csv', 
                                usecols=usecols)
v1_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V1_LO_Front/DET0102_V1_LO_Front.csv',
                                label_path='./Annots/DET0102_V1_LO_Front.csv', 
                                usecols=usecols)
v1_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V1_NO_Front/DET0102_V1_NO_Front.csv',
                                label_path='./Annots/DET0102_V1_NO_Front.csv', 
                                usecols=usecols)


v2_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V2_HI_Front/DET0102_V2_HI_Front.csv',
                                label_path='./Annots/DET0102_V2_HI_Front.csv', 
                                usecols=usecols)
v2_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V2_LO_Front/DET0102_V2_LO_Front.csv',
                                label_path='./Annots/DET0102_V2_LO_Front.csv', 
                                usecols=usecols)
v2_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V2_NO_Front/DET0102_V2_NO_Front.csv',
                                label_path='./Annots/DET0102_V2_NO_Front.csv', 
                                usecols=usecols)


v3_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V3_HI_Front/DET0102_V3_HI_Front.csv',
                                label_path='./Annots/DET0102_V3_HI_Front.csv', 
                                usecols=usecols)
v3_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V3_LO_Front/DET0102_V3_LO_Front.csv',
                                label_path='./Annots/DET0102_V3_LO_Front.csv', 
                                usecols=usecols)
v3_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V3_NO_Front/DET0102_V3_NO_Front.csv',
                                label_path='./Annots/DET0102_V3_NO_Front.csv', 
                                usecols=usecols)


v4_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V4_HI_Front/DET0102_V4_HI_Front.csv',
                                label_path='./Annots/DET0102_V4_HI_Front.csv', 
                                usecols=usecols)
v4_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V4_LO_Front/DET0102_V4_LO_Front.csv',
                                label_path='./Annots/DET0102_V4_LO_Front.csv', 
                                usecols=usecols)
v4_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V4_NO_Front/DET0102_V4_NO_Front.csv',
                                label_path='./Annots/DET0102_V4_NO_Front.csv', 
                                usecols=usecols)


test_name = args[1]
all_video_names = ['v1_hi', 'v1_lo', 'v1_no',
                   'v2_hi', 'v2_lo', 'v2_no',
                   'v3_hi', 'v3_lo', 'v3_no',
                   'v4_hi', 'v4_lo', 'v4_no']
train_video_names = [n for n in all_video_names if test_name not in n]
test_video_names = [n for n in all_video_names if test_name in n]
CAL_MAP_TABLE = {n:eval(n) for n in test_video_names}

if args[7] == 'debug':
    print('debug!')
    train_video_names = train_video_names[:2]
    test_video_names = test_video_names[:1]

print(train_video_names)
print(test_video_names)

CLASSES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
FEATURES = AU_i
# sliding window generate samples
all_train_X = []
all_train_gt_boxes = []
all_train_neg_boxes = []
all_train_gt_boxes_names = []
all_train_global_gt_boxes = {}
for data_name in train_video_names:
    data = eval(data_name)
    train_all_X = data[FEATURES].values
    label_matrix = data[CLASSES].values
    frames = data['frame'].values
    timestamp = data[' timestamp'].values
    
    train_n_samples, train_X, train_gt_boxes, train_neg_boxes = get_feat_and_gt_boxes(
                                                                    train_all_X, label_matrix,
                                                                    training_stride=50, sample_length=416)

    n_samples, train_gt_boxes_names, train_global_gt_boxes = get_global_gt_boxes_and_names(
                                                                    data_name,label_matrix,frames,timestamp,'time',
                                                                    training_stride=50, sample_length=416)
    
    all_train_X.append(train_X)
    all_train_gt_boxes.extend(train_gt_boxes)
    all_train_neg_boxes.extend(train_neg_boxes)
    all_train_gt_boxes_names.extend(train_gt_boxes_names)
    all_train_global_gt_boxes[data_name] = torch.from_numpy(train_global_gt_boxes)
    
    
all_train_X = np.concatenate(all_train_X)

all_train_gt_boxes, all_train_gt_boxes_names = modify_gt_boxes(all_train_gt_boxes, all_train_gt_boxes_names, thresh=3)
all_train_gt_boxes = [torch.from_numpy(e).float() for e in all_train_gt_boxes]
all_train_neg_boxes = modify_gt_boxes(all_train_neg_boxes, thresh=3)
all_train_neg_boxes = [torch.from_numpy(e).float() for e in all_train_neg_boxes]

all_train_X = torch.from_numpy(all_train_X).float()

###
all_test_X = []
all_test_gt_boxes = []
all_test_gt_boxes_names = []
all_test_global_gt_boxes = {}
for data_name in test_video_names:
    data = eval(data_name)
    test_all_X = data[FEATURES].values
    label_matrix = data[CLASSES].values
    frames = data['frame'].values
    timestamp = data[' timestamp'].values

    test_n_samples, test_X, test_gt_boxes, _ = get_feat_and_gt_boxes(
                                                    test_all_X, label_matrix,
                                                    training_stride=50, sample_length=416)

    n_samples, test_gt_boxes_names, test_global_gt_boxes = get_global_gt_boxes_and_names(
                                                                data_name,label_matrix,frames,timestamp,'time',
                                                                training_stride=50, sample_length=416)
    
    all_test_X.append(test_X)
    all_test_gt_boxes.extend(test_gt_boxes)
    all_test_gt_boxes_names.extend(test_gt_boxes_names)
    all_test_global_gt_boxes[data_name] = torch.from_numpy(test_global_gt_boxes)
    
all_test_X = np.concatenate(all_test_X)
all_test_gt_boxes, all_test_gt_boxes_names = modify_gt_boxes(all_test_gt_boxes, all_test_gt_boxes_names, thresh=3)
all_test_gt_boxes = [torch.from_numpy(e).float() for e in all_test_gt_boxes]

all_test_X = torch.from_numpy(all_test_X).float()

# max value normalize 
all_train_X = all_train_X / 5 # max value 5
all_test_X = all_test_X / 5


print('==================')
print('read data done!')
print('==================')
print('Train_X:', all_train_X.shape, 'Test_X:', all_test_X.shape)


BATCH_SIZE = int(args[6])

is_noise = [False, True][1]
train_dataset = TicDataset_all_aug(all_train_X.permute(0,2,1), all_train_gt_boxes, all_train_neg_boxes,
                                    mode='training', 
                                    is_noise=is_noise)
test_dataset = TicDataset_all_aug(all_test_X.permute(0,2,1), all_test_gt_boxes, mode='testing', 
                                    is_noise=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_function, drop_last=False,
                            pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_function, drop_last=False, 
                            pin_memory=True)

print('Train batch_size = {}'.format(BATCH_SIZE))

## anchors
all_proposal_boxes = anchors_det0102()
all_proposal_boxes = all_proposal_boxes.to(device)

# training settings
is_focal = [False,True][0]
net = Det1D(backbone_name='resnet18_large512', device=device,
            thresh_low= -0.4, thresh_high=0.3, loss_type='DetLoss',
            topk_match=8, iou_smooth=False, is_focal=is_focal, uni_mat=False, 
            use_EIoU_loss=True, use_EIoU=True,
            conf_threshold=0.2, nms_threshold=0.2, top_k=100)

net.to(device)

save_name = args[3]
if save_name == 'no_pretrain':
    print('No Pretrained model!')
else:
    pretrained_dict = torch.load('saved_models/{}'.format(save_name), map_location=device)
    net.load_state_dict(pretrained_dict)
    print('load model successful!')
    print(save_name)

if args[7] == 'debug':
    num_epoch = 2
else:
    num_epoch = 200

finetune_or_scratch = args[4]
if finetune_or_scratch == 'finetune':
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-5, betas=(0.843, 0.999), weight_decay=1e-3) # finetune
elif finetune_or_scratch == 'scratch':
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.937, 0.999), weight_decay=1e-3) # scrach

use_scheduler = [False,True][1]
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.8)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=4e-7)

import logging
# log_path = 'log/debug.log'
log_path = args[5]


current_model_name = log_path.split('/')[-1].split('.log')[0]
if not os.path.exists('saved_models/{}'.format(current_model_name)):
    os.makedirs('saved_models/{}'.format(current_model_name))
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file = logging.FileHandler(log_path) # log_path
log_file.setLevel(logging.INFO)
logger.addHandler(log_file)
logger.info('save path: saved_models/{}'.format(current_model_name))

not_use_comet = False
writer = SummaryWriter(comment=log_path.split('/')[-1].strip('.log'), comet_config={"disabled": not_use_comet})


min_loss = 200
last_epoch_loss = 1e-5
early_stop_count = 0
alpha_clf = 1. # 1 # 这个alpha乘在了classification的loss上面
alpha_reg = 2. # 2 # 这个alpha乘在了reg的loss上面
IOU_THRESH = 0.5

scaler = torch.cuda.amp.GradScaler()

num_batch = len(train_dataset) // BATCH_SIZE
print('Training data has {} samples, {}//{}={} batch'.format(len(train_dataset),len(train_dataset),BATCH_SIZE,num_batch))
for epoch in range(num_epoch):
    net.train()
    t_start = time.time()
    
    cur_loss_l,cur_loss_c = 0.0,0.0
    cur_eiou_loss,cur_reg_loss = 0.0,0.0
    for i,(cur_X,cur_gt_boxes,cur_neg_boxes) in enumerate(train_loader):
        cur_X = cur_X.to(device)
        cur_gt_boxes = [e.to(device) for e in cur_gt_boxes]
        cur_neg_boxes = [e.to(device) for e in cur_neg_boxes]
        
        optimizer.zero_grad()       
        with torch.cuda.amp.autocast():  
            clf_loss, reg_loss = net(cur_X, all_proposal_boxes, cur_gt_boxes, cur_neg_boxes)

            rl = alpha_reg*reg_loss.mean()
            clfl = alpha_clf*clf_loss.mean()

            # loss = alpha_reg*reg_loss.mean() + alpha_clf*clf_loss.mean()
            loss = rl + clfl
        
        # loss.backward() ###
        scaler.scale(loss).backward() ###

        ##
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        ##
        # optimizer.step() ###
        scaler.step(optimizer)
        scaler.update()
        
        cur_loss_l += rl.item()
        cur_loss_c += clfl.item()
        cur_eiou_loss += net.eiou_loss.mean().item()
        cur_reg_loss += net.reg_loss.mean().item()

        logger.info('iter {}:({:d}%) | reg_loss:{:.4f} | clf_loss:{:.4f} | total_loss:{:.4f}'.format(
                            i, int(i/num_batch*100), rl.item(), clfl.item(), loss.item()))

    cur_loss_l /= (i+1)
    cur_loss_c /= (i+1)
    cur_loss = cur_loss_l+cur_loss_c
    cur_eiou_loss /= (i+1)
    cur_reg_loss /= (i+1)

    logger.info('epoch:{} | eiou_loss:{:.4f} | reg_loss:{:.4f}'.format(epoch, cur_eiou_loss, cur_reg_loss))

    writer.add_scalars('Train/Loss', {'reg loss': cur_loss_l,
                                      'clf loss': cur_loss_c,
                                      'eiou loss':cur_eiou_loss,
                                      'reg loss alone': cur_reg_loss,
                                      'total loss': cur_loss}, epoch)
    
    if (epoch+1) % 50 == 0:
        print(log_path)
        save_name = '{}_{}_epoch_{}'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),current_model_name,epoch)
        if not os.path.exists('saved_models/{}/'.format(current_model_name)):
            os.makedirs('saved_models/{}/'.format(current_model_name))
        torch.save(net.state_dict(),'saved_models/{}/{}.pth'.format(current_model_name,save_name))
        # eval test
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

        logger.info('epoch:{} | reg_loss:{:.4f} | clf_loss:{:.4f} | test_AP:{:.4f} or {:.4f}'.format(epoch, cur_loss_l, cur_loss_c, test_ap, test_auc))
        writer.add_scalars('Test/metrics', {'AP': test_ap, 'TP#':num_tp, 'FP#':num_fp, 'FN#':num_fn}, epoch)
    else:
        logger.info('epoch:{} | reg_loss:{:.4f} | clf_loss:{:.4f}'.format(epoch, cur_loss_l, cur_loss_c))
    

    if cur_loss < min_loss:
        print('new min loss!')
        save_name = '{}_{}_epoch_{}'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),current_model_name,epoch)
        if not os.path.exists('saved_models/{}/'.format(current_model_name)):
            os.makedirs('saved_models/{}/'.format(current_model_name))
        torch.save(net.state_dict(),'saved_models/{}/{}.pth'.format(current_model_name,save_name))
        # print('model_saved')
        min_loss = cur_loss
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
            print('start eval!')
            try:
                _,_, test_ap,test_auc, TP,MatchedGT,_,_, num_tp, num_fp, num_fn,_,_,_,_,_ = cal_tp_fp_fn(test_video_names,
                                                                                        all_detections,all_test_gt_boxes_names,all_test_global_gt_boxes,
                                                                                        iou_threshold = IOU_THRESH, CAL_MAP_TABLE=CAL_MAP_TABLE)
            except RuntimeError as e:
                print('eval failed!')
                print(e)
            print('end eval!')
        ##------------------------------------------------------------------ eval test
        writer.add_scalars('Test/metrics', {'AP': test_ap, 'TP#':num_tp, 'FP#':num_fp, 'FN#':num_fn}, epoch)
        logger.info('model saved! saved_models/{}/{}.pth. AP:{:.4f} or {:4f}'.format(current_model_name,save_name,test_ap,test_auc))
        
    if cur_loss > last_epoch_loss:
        early_stop_count += 1
    else:
        early_stop_count = 0
    if early_stop_count >= 10:
        print('early stop!')
        writer.close()
        break
    
    if use_scheduler:
        lr_scheduler.step()
    
    last_epoch_loss = cur_loss
    
    if epoch == num_epoch-1:
        print('Last epoch cost {:.2f} s'.format(time.time()-t_start))
    

save_name = '{}_{}_epoch_{}'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),current_model_name,epoch)
torch.save(net.state_dict(),'saved_models/{}/{}.pth'.format(current_model_name,save_name))
print('model_saved: Last model')

writer.close()

### args = sys.argv
# args[0]: it is this file's name
# args[1]: test session, v1
# args[2]: gpu no, 0
# args[3]: pretrained model, 'xx/xx/xx.pth'
# args[4]: finetune or scratch
# args[5]: log path
# args[6]: batch size
# args[7]: debug or not

# python train_det0102_det.py v2 0 no_pretrain scratch log/train_20211229_det0102_[Large512]_[LOSO_v2].log 1024 no_debug

