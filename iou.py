import torch

def center_form(boxes):
    '''
    change boxes in (xmin, xmax) to (cx, w) w是全宽！
    boxes: (n, 2)
    '''
    return torch.stack(((boxes[:,0] + boxes[:,1])/2, boxes[:,1] - boxes[:,0]),1)

def point_form(boxes):
    '''
    change boxes in (cx, w) to (xmin, xmax)
    boxes: (n, 2)
    '''
    return torch.stack((boxes[:,0] - boxes[:,1]/2, boxes[:,0] + boxes[:,1]/2),1)


def intersect(box_a, box_b):
    """
    Input boxes in !!!point_form!!!
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,2].
      box_b: (tensor) bounding boxes, Shape: [B,2].
    Return:
      (tensor) intersection area, Shape: [A,B,1].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 1:].unsqueeze(1).expand(A, B, 1),
                    box_b[:, 1:].unsqueeze(0).expand(A, B, 1))
    min_xy = torch.max(box_a[:, :1].unsqueeze(1).expand(A, B, 1),
                    box_b[:, :1].unsqueeze(0).expand(A, B, 1))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter

def jaccard(box_a, box_b, use_EIoU=False):
    """
    Input boxes in !!!point_form!!!
    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [A,2]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [B,2]
    Return:
        jaccard overlap: (tensor) Shape: [A,B,1]
    """
    A,B = box_a.shape[0], box_b.shape[0]
    
    inter = intersect(box_a, box_b)

    # box_b = torch.clamp(box_b, 0, 415) # clip the proposal

    area_a = (box_a[:,1] - box_a[:,0]).unsqueeze(1).unsqueeze(1).expand_as(inter)
    area_b = (box_b[:,1] - box_b[:,0]).unsqueeze(1).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    union = torch.clamp(union, min=1e-8)
    iou = inter / union
    
    if use_EIoU:
        # central point dist
        d = torch.abs(((box_b[:,0]+box_b[:,1])/2).repeat(A,1) - ((box_a[:,0]+box_a[:,1])/2).repeat(B,1).t()) # (A,B)
        c = torch.max(box_b[:,1].repeat(A,1), box_a[:,1].repeat(B,1).t()) - torch.min(box_b[:,0].repeat(A,1), box_a[:,0].repeat(B,1).t())
        assert torch.all(c>=0), print(box_b,'\n\n\n\n',box_a)
        cpd = d**2 / c**2

        # width dist
        w_diff = torch.abs((box_b[:,1]-box_b[:,0]).repeat(A,1) - (box_a[:,1]-box_a[:,0]).repeat(B,1).t())
        wd = w_diff**2 / c**2

        eiou = iou[:,:,0] - cpd - wd
        return eiou.unsqueeze(-1)
    else:
        return iou  # [A,B,1]

def cal_IoG(box_a, box_b):
    inter = torch.diag(intersect(box_a,box_b)[:,:,0])
    area_b = box_b[:,1]-box_b[:,0]
    iog = inter / torch.clamp(area_b, min=1e-8)
    return iog
  


def matching_strategy(box_a, box_b, gt_labels, device):
    '''
    box_a: GT boxes, (A,2)
    box_b: proposal boxes, (B,2)
    Input boxes in !!!point_form!!!

    Output: index corresponding to which GT box, (B,1)
    '''
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    iou = jaccard(box_a, box_b) # gt -> proposal boxes
    # iou_pg = iou.permute(1,0,2) # proposal boxes -> gt
    gt_to_pps_iou, gt_to_pps_ind = iou.max(dim=1) # ground truth corresponds to which proposal box (max IOU)

    pps_to_gt_iou, pps_to_gt_ind = iou.max(dim=0) # proposal box corresponds to which ground truth (max IOU)
    mask = pps_to_gt_iou >= 0.3
    pps_to_gt_ind = pps_to_gt_ind * mask + ~mask * (-1) # only keep those "proposal box corresponds to ground truth" with IOU > 0.5
    pps_to_gt_ind[gt_to_pps_ind.squeeze()] = torch.arange(0,box_a.shape[0]).view(-1,1).to(device)
    pps_to_gt_ind.squeeze_()

    # proposal_labels = gt_labels[pps_to_gt_ind]
    # proposal_labels[pps_to_gt_ind < 0] = 0
    proposal_labels = torch.zeros(pps_to_gt_ind.shape[0]).long()
    proposal_labels[pps_to_gt_ind>-1]=1
    proposal_labels = proposal_labels.long().to(device)

    proposal_corresponding_gt_boxes = box_a[pps_to_gt_ind]
    proposal_corresponding_gt_boxes[pps_to_gt_ind < 0] = float('inf')

    return proposal_labels, proposal_corresponding_gt_boxes


