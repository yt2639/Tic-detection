import torch
import torch.nn as nn
from loss import BBoxTransform1D, ClipBoxes1D
from nms import nms

class Detection1D(nn.Module):
    '''
    used for inference, detecting boxes
    '''
    def __init__(self,
                conf_threshold = 0.01,
                nms_threshold = 0.5,
                top_k = 10,
                use_EIoU=False, num_classes=1):
        super(Detection1D, self).__init__()
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.use_EIoU = use_EIoU
        self.apply_offset = BBoxTransform1D()
        self.clip_boundary = ClipBoxes1D()

    def forward(self, clf_proba, reg_preds_all, all_proposal_boxes, device):
        length_thresh = 3

        with torch.no_grad():
        ### 
            pred_boxes = self.apply_offset(all_proposal_boxes, reg_preds_all)
            pred_boxes = self.clip_boundary(pred_boxes, boundary=[0,416])

            self.ori_det = torch.cat((pred_boxes.clone(),clf_proba.clone()),-1)

            detections = -torch.ones(clf_proba.size(0), self.top_k, 3) # (batch_size, topk, 3)
            detections = detections.to(device)
            ####
            for cl in range(clf_proba.size(2)): # cl = 0
                for i in range(clf_proba.size(0)):
                    c_mask = clf_proba[i,:,cl].gt(self.conf_threshold)
                    cur_score = clf_proba[i,c_mask,cl:cl+1] # (num_anchor, 1)
                    if cur_score.size(0) == 0:
                        continue 
                    cur_pred_boxes_point_form = pred_boxes[i, c_mask] # (num_anchor, 2)
                    # delete very small boxes
                    mask = (cur_pred_boxes_point_form[:,1] - cur_pred_boxes_point_form[:,0]) > length_thresh
                    cur_pred_boxes_point_form = cur_pred_boxes_point_form[mask,:]
                    cur_score = cur_score[mask]
                    
                    idx, count = nms(cur_pred_boxes_point_form, cur_score, nms_threshold=self.nms_threshold, top_k=self.top_k, use_EIoU=self.use_EIoU)
                    
                    results = torch.cat((cur_pred_boxes_point_form[idx], cur_score[idx]),1) # (topk, 3), x,y,p
                    # results = results.sort(-1,descending=True)[0]
                    
                    detections[i,:count,:] = results
                    
        return detections
    

