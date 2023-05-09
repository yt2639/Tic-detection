import torch
import numpy as np
import sklearn
from config import config


def GLOBAL_cal_seg_metrics(all_seg_results, all_test_video_general_labels, all_test_gt_boxes_names, 
                            SEG_TEST_PREDEFINE, test_video_names,
                            p=2, smooth=1e-3, thresh=0.5,
                            sample_stride=50, sample_length=416):
    all_seg = {}
    for n in test_video_names:
        pre_defined = SEG_TEST_PREDEFINE[n]
        all_seg[n] = torch.zeros(pre_defined[0],pre_defined[1])
    # # ------------------------------------------------------------------ 创建result矩阵


    for i in range(len(all_test_gt_boxes_names)):
        i_th, v_name = all_test_gt_boxes_names[i].dtype.names[0].split('-')
        if i_th == 'final':
            all_seg[v_name][-1, -config['sample_length']:] = all_seg_results[i].squeeze()
        else:
            i_th = int(i_th)
            all_seg[v_name][i_th, i_th*sample_stride:(i_th*sample_stride+sample_length)] = all_seg_results[i].squeeze()
            
    for k in all_seg.keys():
        all_seg[k] = all_seg[k].sum(0) / torch.clamp((all_seg[k] != 0.).sum(0), min=1)


    all_acc,all_recall,all_precision,all_f1,all_dice = [],[],[],[],[]
    all_auc = []
    for b,v_name in enumerate(test_video_names):

        proba = all_seg[v_name]
        preds = (proba >= thresh) * 1.

        proba = proba.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        
        lbs = all_test_video_general_labels[v_name].squeeze()#.detach().cpu().numpy()

        # cal metrics
        # print(preds.shape,type(preds),lbs.shape,type(lbs))
        acc = np.mean(((preds == lbs)*1.0))
        recall = sklearn.metrics.recall_score(lbs, preds)
        precision = sklearn.metrics.precision_score(lbs, preds)
        f1 = sklearn.metrics.f1_score(lbs, preds)

        # roc-auc
        fpr, tpr, _ = sklearn.metrics.roc_curve(lbs, proba)
        auc = sklearn.metrics.auc(fpr, tpr)

        # dice
        intersection = (proba * lbs)
        if p==2:
            dice = (2.*intersection.sum() + smooth) / ((proba**2).sum() + (lbs**2).sum() + smooth)
        elif p==1:
            dice = (2.*intersection.sum() + smooth) / (proba.sum() + lbs.sum() + smooth)
    
        all_acc.append(acc)
        all_recall.append(recall)
        all_precision.append(precision)
        all_f1.append(f1)
        all_dice.append(dice)
        all_auc.append(auc)
        
    all_acc = np.mean(all_acc)
    all_recall = np.mean(all_recall)
    all_precision = np.mean(all_precision)
    all_f1 = np.mean(all_f1)
    all_dice = np.mean(all_dice)
    all_auc = np.mean(all_auc)
    
    return all_dice,all_acc,all_recall,all_precision,all_f1,all_auc



