from iou import jaccard
import numpy as np
import torch

def nms(cur_pred_boxes_point_form, cur_score, nms_threshold=0.5, top_k=10, use_EIoU=False):
    '''
    This function is NMS. It's used for ONE class and ONE person! So you need two for loops in the "eval.py" code
    cur_pred_boxes_point_form: (num_anchor, 2)
    cur_score: (num_anchor, 1)
    '''
    keep = cur_score.new(cur_score.size(0)).zero_().long()
    _, idx = cur_score.squeeze().sort(descending=False)

    count = 0
    while idx.numel() > 0: 
        if not idx.shape: 
            break
        # print(idx.shape)
        largest = idx[-1].item()
        keep[count] = largest
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1] # delete largest score's index
        
        #
        largest_box = cur_pred_boxes_point_form[largest]
        other_boxes = cur_pred_boxes_point_form[idx]
        IoU = jaccard(largest_box.reshape(-1,2), other_boxes.reshape(-1,2), use_EIoU=use_EIoU).squeeze()
        idx = idx[IoU.le(nms_threshold)]
    
    if count > top_k:
        count = top_k

    return keep[:count], count

