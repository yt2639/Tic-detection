import torch
import torch.nn as nn
import torch.nn.functional as F
# import kornia

from iou import center_form, jaccard
import random
# import cvxpy as cp
import warnings
warnings.filterwarnings('ignore')


def apply_offset(os_cx, os_w, proposal_boxes):
    '''
    Input and Output proposal_boxes: in center_form !!!!
    os: offset
    '''
    # with torch.no_grad():
    new_cx = os_cx * proposal_boxes[:,1] + proposal_boxes[:,0]
    new_w = torch.exp(os_w) * proposal_boxes[:,1]
    new_proposal_boxes = torch.stack((new_cx,new_w),dim=-1)

    return new_proposal_boxes

def log_sum_exp(x): 
    """
    This is only the addition for the ** denominator **.
    We use this function to calculate cross_entropy_loss for negative samples. 
    cross_entropy_loss has two components: loss(p,class) = −log( exp(p[class]) / ∑j exp(p[j]) )
                                                         = −p[class] + log( ∑j exp(p[j]) )
    Args:
        x: net output, confidence, (batch_size * num_anchor_boxes, num_classes)
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

class BBoxTransform1D(nn.Module):

    def __init__(self):
        super(BBoxTransform1D, self).__init__()

    def forward(self, boxes, reg_preds):

        widths  = boxes[:, 1] - boxes[:, 0]
        ctr_x   = boxes[:, 0] + 0.5 * widths

        dx = reg_preds[:, :, 0] * 0.1
        dw = reg_preds[:, :, 1] * 0.2

        pred_ctr_x = ctr_x + dx * widths
        pred_w     = torch.exp(dw) * widths

        pred_boxes_xmin = pred_ctr_x - 0.5 * pred_w
        pred_boxes_xmax = pred_ctr_x + 0.5 * pred_w

        pred_boxes = torch.stack([pred_boxes_xmin, pred_boxes_xmax], dim=2)

        return pred_boxes

class ClipBoxes1D(nn.Module):

    def __init__(self):
        super(ClipBoxes1D, self).__init__()

    def forward(self, boxes, boundary=[0,416]):

        xmin, xmax = boundary

        # boxes[:, 0] = torch.clamp(boxes[:, 0], min=xmin)
        # boxes[:, 1] = torch.clamp(boxes[:, 1], max=xmax)
        
        boxes = torch.clamp(boxes, min=xmin, max=xmax)
      
        return boxes#.clone()

def EIoU_loss(pred_boxes, target_boxes):
    '''
    Input is in point_form!!!!!!
    '''
    # calculate IoU
    # pos_IoU = torch.diag(jaccard(pred_boxes, target_boxes, use_EIoU=False)[:,:,0])
    pos_IoU = []
    for i in range(pred_boxes.shape[0]):
        pos_IoU.append(jaccard(pred_boxes[i:i+1], target_boxes[i:i+1], use_EIoU=False).reshape(-1,))
    
    pos_IoU = torch.cat(pos_IoU)

    # central point dist
    d = torch.abs((pred_boxes[:,0]+pred_boxes[:,1])/2 - (target_boxes[:,0]+target_boxes[:,1])/2)
    c = torch.max(pred_boxes[:,1], target_boxes[:,1]) - torch.min(pred_boxes[:,0], target_boxes[:,0])
    assert torch.all(c>=0), print(c)
    cpd = d**2 / torch.maximum(c**2, torch.tensor([1e-6]).to(d.device))

    # width dist
    w_diff = torch.abs((pred_boxes[:,1]-pred_boxes[:,0]) - (target_boxes[:,1]-target_boxes[:,0]))
    wd = w_diff**2 / torch.maximum(c**2, torch.tensor([1e-6]).to(d.device))

    eiou = pos_IoU - cpd - wd
    loss = 1 - eiou
    return loss

class DetLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, gt_boxes, neg_boxes=None, device='cpu',
                neg_pos_ratio=3, thresh_low=0.03, thresh_high=0.3, topk_match=8, iou_smooth=False, 
                is_focal=False, uni_mat=True, use_EIoU_loss=False, use_MAE_proba_loss=False, use_EIoU=False, **kwargs):
        '''
        anchors: point_form!!
        ignore between thresh_low and thresh_high
        classifications: (N,...,1)
        regressions: (N,...,2)
        '''
        assert neg_boxes is not None
        neg_anchor_thresh = 0.75

        alpha = 0.25
        gamma = 2.0
        if is_focal:
            beta = 1.0 / 9.0
            # beta = 1.0
        else:
            beta = 1.0
        num_hard = 50 # 80, 30
        num_neg_random = 600 # randomly select num_neg_random from neg loss
        # neg_pos_ratio = 2 # 3
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        self.EIoU_losses = []
        self.reg_losses = []
        alpha_reg = 1
        alpha_eiou = 1.5

        for i in range(batch_size):
            classification = classifications[i]
            regression = regressions[i]
            gt_annotation = gt_boxes[i]
            neg_annotation = neg_boxes[i]

            # gt_annotation = gt_annotation[gt_annotation[:, 4] != -1]
            targets = torch.zeros(classification.shape).to(device)
            # print('targets:',targets.shape)

            if gt_annotation.shape[0] > 0:
                iou = jaccard(anchors, gt_annotation, use_EIoU=use_EIoU)[:,:,0] # (num_anchor, num_gt) (130,5)

                # neg box matching
                neg_ind = None
                if len(neg_annotation) > 0:
                    neg_iou = jaccard(anchors, neg_annotation, use_EIoU=False)[:,:,0]
                    neg_ind = (neg_iou > neg_anchor_thresh).sum(1) > 0 # [1248], 1d
                    iou[neg_ind] = -1

                # matching strategy
                IoU_max_values, IoU_argmax = iou.max(dim=1) # num_anchor x 1
                # print(IoU_max_values.shape)
                
                assigned_annotations = gt_annotation[IoU_argmax] # (130,2)
                # targets[IoU_max_values >= 0.5] = 1
                targets[torch.ge(IoU_max_values,thresh_low) & torch.lt(IoU_max_values,thresh_high)] = -1.0 # ignore

                pos_ind = torch.ge(IoU_max_values, thresh_high)#.bool()

                num_pos = pos_ind.sum()
                # one hot
                targets[pos_ind] = 1.0
                
                if neg_ind is not None:
                    targets[neg_ind] = 99

                ori_targets = targets.clone()

                if neg_ind is not None:
                    targets[neg_ind] = 0
            else:
                # neg box matching
                neg_ind = None
                if len(neg_annotation) > 0:
                    neg_iou = jaccard(anchors, neg_annotation, use_EIoU=False)[:,:,0]
                    neg_ind = (neg_iou > neg_anchor_thresh).sum(1) > 0
                    targets[neg_ind] = 99
                
                ori_targets = targets.clone()

                if neg_ind is not None:
                    targets[neg_ind] = 0

                pos_ind = torch.tensor(0).to(device)
                num_pos = torch.tensor(0).to(device)


            bce = F.binary_cross_entropy_with_logits(classification, targets, reduce=False)
            if is_focal:
                # 这是原来手动的做法
                clf_prob = torch.clamp(torch.sigmoid(classification), 1e-4, 1.0-1e-4)

                alpha_factor = torch.ones(ori_targets.shape) * alpha
                alpha_factor = alpha_factor.to(device)
                alpha_factor = torch.where(torch.eq(ori_targets, 1.), alpha_factor, 1. - alpha_factor)
                focal_weight = torch.where(torch.eq(ori_targets, 1.), 1. - clf_prob, clf_prob)
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                
                clf_loss = focal_weight * bce

                # clf_loss = kornia.losses.binary_focal_loss_with_logits(classification.unsqueeze(0), targets.unsqueeze(0), 
                #                                                        alpha=alpha, gamma=gamma, reduction='none').squeeze(0)
            else:
                clf_loss = bce
            
            clf_loss = torch.where(torch.ne(ori_targets, -1.0), clf_loss, torch.zeros(clf_loss.shape).to(device))
            # print(clf_loss.shape)
            # print(clf_loss)
            if num_pos > 0:
                neg_loss = clf_loss[ori_targets == 0]
                really_neg_loss = clf_loss[ori_targets == 99]
                pos_loss = clf_loss[ori_targets == 1]
                # print(neg_loss.shape)
                if not is_focal:
                    # hard negative mining
                    neg_idcs = random.sample(range(len(neg_loss)), min(num_neg_random, len(neg_loss)))
                    neg_loss = neg_loss[neg_idcs]

                    # print(neg_pos_ratio * num_pos, neg_loss.size(0))
                    # assert neg_pos_ratio * num_pos <= neg_loss.size(0)

                    if neg_pos_ratio * num_pos > neg_loss.size(0):
                        num_neg = neg_loss.size(0)
                    else:
                        num_neg = neg_pos_ratio * num_pos
                    _, hard_neg_ind = torch.topk(neg_loss, num_neg)
                    neg_loss = neg_loss[hard_neg_ind]

                clf_loss = pos_loss.sum() + neg_loss.sum() + really_neg_loss.sum()

            else: # no gt boxes
                neg_loss = clf_loss[ori_targets == 0]
                really_neg_loss = clf_loss[ori_targets == 99]
                if not is_focal:
                    # hard negative mining
                    neg_idcs = random.sample(range(len(neg_loss)), min(num_neg_random, len(neg_loss)))
                    neg_loss = neg_loss[neg_idcs]

                    assert len(neg_loss) > num_hard
                    _, hard_neg_ind = torch.topk(neg_loss, num_hard) # num_hard=30
                    neg_loss = neg_loss[hard_neg_ind]

                clf_loss = neg_loss.sum() + really_neg_loss.sum()

            classification_losses.append(clf_loss.sum() / torch.clamp(num_pos, min=1))

            # reg loss
            if num_pos > 0:
                assigned_annotations = assigned_annotations[pos_ind, :]
                gt_center_form = center_form(assigned_annotations)
                gt_cx = gt_center_form[:,0]
                gt_w = gt_center_form[:,1]

                if use_EIoU_loss:
                    pred_boxes = BBoxTransform1D()(anchors, regression.unsqueeze(0)) # (1, 1248, 2)
                    pred_boxes = pred_boxes[:,pos_ind]
                    pred_boxes = torch.clamp(pred_boxes, min=0, max=416) # (1, 1248, 2)

                    eiou_loss = EIoU_loss(pred_boxes.squeeze(0), assigned_annotations)
                    self.EIoU_losses.append(eiou_loss.mean())
                
                anchors_center_form = center_form(anchors[pos_ind])
                anchor_cx = anchors_center_form[:,0]
                anchor_w = anchors_center_form[:,1]

                targets_dcx = (gt_cx - anchor_cx) / anchor_w
                targets_dw = torch.log(gt_w / anchor_w) 

                targets_reg = torch.stack((targets_dcx,targets_dw))
                targets_reg = targets_reg.t()
                targets_reg = targets_reg / torch.Tensor([[0.1, 0.2]]).to(device)
                regression_diff = torch.abs(targets_reg - regression[pos_ind, :])
                reg_loss = torch.where(
                    torch.le(regression_diff, beta),
                    0.5 * torch.pow(regression_diff, 2) / beta,
                    regression_diff - 0.5 * beta
                )
                self.reg_losses.append(reg_loss.mean())
                
                if use_EIoU_loss:
                    regression_losses.append(alpha_reg*reg_loss.mean() + alpha_eiou*eiou_loss.mean())
                else:
                    regression_losses.append(reg_loss.mean())

            else:
                regression_losses.append(torch.tensor(0.0).float().to(device))
        
        self.EIoU_losses = alpha_eiou*torch.stack(self.EIoU_losses).mean(dim=0, keepdim=True)
        self.reg_losses = alpha_reg*torch.stack(self.reg_losses).mean(dim=0, keepdim=True)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)



