import numpy as np
import cv2
from torch.utils.data import Dataset
import torch

TEST_TIC_IF_INCLUDE_13_14 = 13 # class 13 14 are vocal tics

def get_start_and_end(label_sequence_uint8):
    '''
    "label_sequence_uint8" should be a np.array, (seq_length, 1), uint8
    '''
    a=cv2.connectedComponents(label_sequence_uint8)
    seg = a[1].squeeze()
    tic_segment = [] # Ground Truth
    # labels = []
    for i in range(1,a[0]):
        sequence = (seg == i).nonzero()[0]
        if len(sequence) == 1:
            pass
        else:
            tic_segment.append([sequence.min(),sequence.max()+1]) # sequence.max()+1
        
    return np.array(tic_segment)

def get_start_and_end_np_unique(label_sequence):
    unique_values = np.unique(label_sequence)
    tic_segment = [] # Ground Truth
    if len(unique_values) > 0:
        for uv in unique_values[1:]:
            sequence = (label_sequence == uv).nonzero()[0]
            if len(sequence) == 1:
                pass
            else:
                tic_segment.append([sequence.min(),sequence.max()+1]) # sequence.max()+1

    return np.array(tic_segment)
    
def modify_gt_boxes(gt_boxes, gt_boxes_names=None, mode='delete_small', thresh=3):
    '''
    if the gt box is too small, then it probably is an annotation error which should be deleted.
    '''
    new_gt_boxes_delete_lt_3 = []
    new_gt_boxes_names_delete_lt_3 = []
    for i in range(len(gt_boxes)):
        new_e = gt_boxes[i]
        if gt_boxes_names is not None:
            new_name = gt_boxes_names[i]
        if len(new_e)>0:
            LENGTH = new_e[:,1] - new_e[:,0]
            if mode == 'delete_small':
                new_e = new_e[LENGTH > thresh] # thresh=3
                if gt_boxes_names is not None:
                    new_name = new_name[LENGTH > thresh]
            elif mode == 'delete_large':
                new_e = new_e[LENGTH < thresh] # thresh=3
                if gt_boxes_names is not None:
                    new_name = new_name[LENGTH < thresh]

        new_gt_boxes_delete_lt_3.append(new_e)
        if gt_boxes_names is not None:
            new_gt_boxes_names_delete_lt_3.append(new_name)
    
    if gt_boxes_names is not None:
        return new_gt_boxes_delete_lt_3, new_gt_boxes_names_delete_lt_3
    else:
        return new_gt_boxes_delete_lt_3

def delete_empty_gt(X, gt_boxes):
    m = [i for i,e in enumerate(gt_boxes) if len(e)>0]
    gt_boxes_delete_empty = [e for e in gt_boxes if len(e)>0]
    X_delete_empty = X[m]
    return X_delete_empty, gt_boxes_delete_empty



def get_feat_and_gt_boxes(X, label_matrix,
                        training_stride=50, sample_length=416,
                        ):
    ''' Use overlapping window and use the "stride" to control how many samples you want to generate.
    X: the whole feature matrix, (N,feature_dim)
    label_matrix: matrix for all 15 class
    training_stride: stride for the sliding window
    sample_length: length of the sliding window
    '''
    all_X = []
    all_gt_boxes = []
    all_neg_boxes = []   

    n_samples = (X.shape[0]-sample_length) // training_stride
    for i in range(n_samples+1):
        cur_X = X[i*training_stride:(i*training_stride+sample_length)]
        all_X.append(cur_X)
        
        gt_boxes = np.empty(shape=[0, 2], dtype=np.int32)
        for tic_class in range(4, TEST_TIC_IF_INCLUDE_13_14):
            y = label_matrix[i*training_stride:(i*training_stride+sample_length), tic_class]
            one_gt_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
            if one_gt_boxes.shape[0] > 0:
                gt_boxes = np.concatenate((gt_boxes, one_gt_boxes))
        all_gt_boxes.append(gt_boxes)
        
        y = label_matrix[i*training_stride:(i*training_stride+sample_length), 1] # 1!!
        neg_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
        all_neg_boxes.append(neg_boxes)

    ###
    cur_X = X[-sample_length:]
    all_X.append(cur_X)
    
    gt_boxes = np.empty(shape=[0, 2], dtype=np.int32)
    for tic_class in range(4, TEST_TIC_IF_INCLUDE_13_14):
        y = label_matrix[-sample_length:, tic_class]
        one_gt_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
        if one_gt_boxes.shape[0] > 0:
            gt_boxes = np.concatenate((gt_boxes, one_gt_boxes))
    all_gt_boxes.append(gt_boxes)

    y = label_matrix[-sample_length:, 1] # 1!!
    neg_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
    all_neg_boxes.append(neg_boxes)

    return n_samples+2, np.stack(all_X), all_gt_boxes, all_neg_boxes

def get_global_gt_boxes_and_names(name, label_matrix, t_frame, t_time, name_mode='time',
                                training_stride=50, sample_length=416,
                                ):
    '''
    similar to "get_feat_and_gt_boxes", but get global gt boxes and names for testing and debugging
    '''
    if name_mode == 'time':
        frames = t_time
    if name_mode == 'frame':
        frames = t_frame

    gt_boxes = np.empty(shape=[0, 2], dtype=np.int32)
    for tic_class in range(4,TEST_TIC_IF_INCLUDE_13_14):
        y = label_matrix[:, tic_class]
        one_gt_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
        if one_gt_boxes.shape[0] > 0:
            gt_boxes = np.concatenate((gt_boxes, one_gt_boxes))

    all_gt_boxes_names = []
    n_samples = (label_matrix.shape[0]-sample_length) // training_stride
    for i in range(n_samples):
        cur_frames = frames[i*training_stride:(i*training_stride+sample_length)]
        gt_boxes_names = np.array([])
        for tic_class in range(4,TEST_TIC_IF_INCLUDE_13_14):
            y = label_matrix[i*training_stride:(i*training_stride+sample_length), tic_class]
            one_gt_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
            if one_gt_boxes.shape[0] > 0:
                # record the name
                for b in one_gt_boxes:
                    cur_name = name + '-'+str(cur_frames[b[0]]) + '-'+str(cur_frames[b[1]-1])
                    gt_boxes_names = np.concatenate((gt_boxes_names, np.array([cur_name])))
        
        final_name = gt_boxes_names.reshape(-1,1)
        final_name = np.array(final_name, dtype=[(str(i)+'-'+name, '<U33')])
        all_gt_boxes_names.append(final_name)
         
    # 
    cur_frames = frames[(i+1)*training_stride:((i+1)*training_stride+sample_length)]
    gt_boxes_names = np.array([])
    for tic_class in range(4,TEST_TIC_IF_INCLUDE_13_14):
        y = label_matrix[(i+1)*training_stride:((i+1)*training_stride+sample_length), tic_class]
        one_gt_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
        if one_gt_boxes.shape[0] > 0:
            # record the name
            for b in one_gt_boxes:
                cur_name = name + '-'+str(cur_frames[b[0]]) + '-'+str(cur_frames[b[1]-1])
                gt_boxes_names = np.concatenate((gt_boxes_names, np.array([cur_name])))
    
    final_name = gt_boxes_names.reshape(-1,1)
    final_name = np.array(final_name, dtype=[(str(i+1)+'-'+name, '<U33')])
    all_gt_boxes_names.append(final_name)

    ###
    cur_frames = frames[-sample_length:]
    gt_boxes_names = np.array([])
    for tic_class in range(4,TEST_TIC_IF_INCLUDE_13_14):
        y = label_matrix[-sample_length:, tic_class]
        one_gt_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
        if one_gt_boxes.shape[0] > 0:
            # record the name
            for b in one_gt_boxes:
                cur_name = name + '-'+str(cur_frames[b[0]]) + '-'+str(cur_frames[b[1]-1])
                gt_boxes_names = np.concatenate((gt_boxes_names, np.array([cur_name])))
    
    final_name = gt_boxes_names.reshape(-1,1)
    final_name = np.array(final_name, dtype=[('final'+'-'+name, '<U33')])
    all_gt_boxes_names.append(final_name)

    return n_samples+2, all_gt_boxes_names, gt_boxes

def get_feat_and_gt_boxes_with_general_labels(
                                            X, label_matrix, general_labels,
                                            training_stride=50, sample_length=416,
                                            ):
    all_X = []
    all_gt_boxes = []
    all_neg_boxes = []   
    all_general_labels = []

    n_samples = (X.shape[0]-sample_length) // training_stride
    for i in range(n_samples+1):
        cur_X = X[i*training_stride:(i*training_stride+sample_length)]
        all_X.append(cur_X)

        cur_general_labels = general_labels[i*training_stride:(i*training_stride+sample_length)]
        all_general_labels.append(cur_general_labels)
        
        gt_boxes = np.empty(shape=[0, 2], dtype=np.int32)
        for tic_class in range(4,TEST_TIC_IF_INCLUDE_13_14):
            y = label_matrix[i*training_stride:(i*training_stride+sample_length), tic_class]
            one_gt_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
            if one_gt_boxes.shape[0] > 0:
                gt_boxes = np.concatenate((gt_boxes, one_gt_boxes))
        all_gt_boxes.append(gt_boxes)
        
        y = label_matrix[i*training_stride:(i*training_stride+sample_length), 1] # 1!!
        neg_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
        all_neg_boxes.append(neg_boxes)

    ###
    cur_X = X[-sample_length:]
    all_X.append(cur_X)

    cur_general_labels = general_labels[-sample_length:]
    all_general_labels.append(cur_general_labels)
    
    gt_boxes = np.empty(shape=[0, 2], dtype=np.int32)
    for tic_class in range(4,TEST_TIC_IF_INCLUDE_13_14):
        y = label_matrix[-sample_length:, tic_class]
        one_gt_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
        if one_gt_boxes.shape[0] > 0:
            gt_boxes = np.concatenate((gt_boxes, one_gt_boxes))
    all_gt_boxes.append(gt_boxes)

    y = label_matrix[-sample_length:, 1] # 1!!
    neg_boxes = get_start_and_end(y.reshape(1,-1).astype(np.uint8))
    all_neg_boxes.append(neg_boxes)

    return n_samples+2, np.stack(all_X), np.stack(all_general_labels), all_gt_boxes, all_neg_boxes



### dataset
def collate_function(data):
    """
    :data: a list for a batch of samples. [[string, tensor], ..., [string, tensor]]
    """
    transposed_data = list(zip(*data))
    gt_boxes = []
    for ele in transposed_data[1]:
        if isinstance(ele, list):
            gt_boxes.extend(ele)
        else:
            gt_boxes.append(ele)
    X = torch.cat(transposed_data[0])

    if len(transposed_data) == 3: # neg box
        neg_boxes = []
        for ele in transposed_data[2]:
            if isinstance(ele, list):
                neg_boxes.extend(ele)
            else:
                neg_boxes.append(ele)
        return (X, gt_boxes, neg_boxes)

    if len(transposed_data) == 4: # neg box，ignore box
        neg_boxes = []
        for ele in transposed_data[2]:
            if isinstance(ele, list):
                neg_boxes.extend(ele)
            else:
                neg_boxes.append(ele)

        ignore_boxes = []
        for ele in transposed_data[3]:
            if isinstance(ele, list):
                ignore_boxes.extend(ele)
            else:
                ignore_boxes.append(ele)
        return (X, gt_boxes, neg_boxes, ignore_boxes)

    return (X, gt_boxes)#, index)

class TicDataset_all_aug(Dataset):
    def __init__(self, X, gt_boxes, neg_boxes=None, ignore_boxes=None, 
                 mode='training',
                 is_noise=False):
        self.mode = mode
        
        self.is_noise = is_noise

        # self.X = torch.from_numpy(X).float()
        self.X = X
        self.gt_boxes = gt_boxes
        self.neg_boxes = neg_boxes
        self.ignore_boxes = ignore_boxes
    
    def __getitem__(self,idx,split=None):
        aug_times = 3

        all_X = []
        all_gt_boxes = []
        all_neg_boxes = []
        all_ignore_boxes = []

        if self.mode == 'training':
            rd_noise = 0
            all_X.append(self.X[idx]) # .unsqueeze(0)
            # all_gt_boxes.append(self.gt_boxes[idx])
            
            if self.is_noise:
                rd_noise = torch.randint(0,aug_times,(1,))
                # print(rd)
                if rd_noise > 0:
                    X = self.X[idx].clone()
                    for _ in range(rd_noise):
                        X_noise = X + torch.normal(0.0,0.01,X.shape)
                        X_noise = torch.clamp(X_noise, min=0.0, max=1.0)
                        all_X.append(X_noise)

            all_gt_boxes.extend([self.gt_boxes[idx]]*(1+rd_noise))
            all_X = torch.stack(all_X) 

            if self.neg_boxes is not None:
                all_neg_boxes.extend([self.neg_boxes[idx]]*(1+rd_noise))
                if self.ignore_boxes is not None:
                    all_ignore_boxes.extend([self.ignore_boxes[idx]]*(1+rd_noise))
                    return all_X, all_gt_boxes, all_neg_boxes, all_ignore_boxes
                else:
                    return all_X, all_gt_boxes, all_neg_boxes
            else:
                return all_X, all_gt_boxes

        else:
            all_X.append(self.X[idx])
            all_X = torch.stack(all_X) 
            all_gt_boxes.append(self.gt_boxes[idx])
            # if self.neg_boxes is not None:
            #     all_neg_boxes.append(self.neg_boxes[idx])
            #     return all_X, all_gt_boxes, all_neg_boxes
            # else:
            return all_X, all_gt_boxes
    
    def __len__(self):
        return self.X.shape[0]

def collate_function_with_general_labels(data):
    """
    :data: a list for a batch of samples. [[string, tensor], ..., [string, tensor]]
    """
    transposed_data = list(zip(*data))

    X = torch.cat(transposed_data[0])

    general_labels = torch.cat(transposed_data[1])

    gt_boxes = []
    for ele in transposed_data[2]:
        if isinstance(ele, list):
            gt_boxes.extend(ele)
        else:
            gt_boxes.append(ele)
    
    if len(transposed_data) == 4: 
        neg_boxes = []
        for ele in transposed_data[3]:
            if isinstance(ele, list):
                neg_boxes.extend(ele)
            else:
                neg_boxes.append(ele)
        
        return (X, general_labels, gt_boxes, neg_boxes)

    else:
        return (X, general_labels, gt_boxes)

class TicDataset_all_aug_with_general_labels(Dataset):
    def __init__(self, X, general_labels, gt_boxes, neg_boxes=None, ignore_boxes=None, 
                 mode='training',
                 is_noise=False):
        self.mode = mode
        
        self.is_noise = is_noise

        # self.X = torch.from_numpy(X).float()
        self.X = X
        self.general_labels = general_labels
        self.gt_boxes = gt_boxes
        self.neg_boxes = neg_boxes
        self.ignore_boxes = ignore_boxes
    
    def __getitem__(self,idx,split=None):
        all_X = []
        all_general_labels = []
        all_gt_boxes = []
        all_neg_boxes = []

        if self.mode == 'training':
            rd_noise = 0
            all_X.append(self.X[idx]) # .unsqueeze(0)
            all_general_labels.append(self.general_labels[idx])
            
            if self.is_noise:
                # rd_noise = torch.randint(0,aug_times,(1,))
                rd_noise = 1
                # print(rd)
                if rd_noise > 0:
                    X = self.X[idx].clone() # [C=17, T=416]
                    for _ in range(rd_noise):
                        X_noise = X + torch.normal(0.0,0.01,X.shape)
                        X_noise = torch.clamp(X_noise, min=0.0, max=1.0)
                        all_X.append(X_noise)
                        all_general_labels.append(self.general_labels[idx])

            all_X = torch.stack(all_X) 
            all_general_labels = torch.stack(all_general_labels)
            all_gt_boxes.extend([self.gt_boxes[idx]]*(1+rd_noise))
            all_neg_boxes.extend([self.neg_boxes[idx]]*(1+rd_noise))

            return all_X, all_general_labels, all_gt_boxes, all_neg_boxes

        else:
            all_X.append(self.X[idx])
            all_X = torch.stack(all_X) 
            all_general_labels.append(self.general_labels[idx])
            all_general_labels = torch.stack(all_general_labels)
            all_gt_boxes.append(self.gt_boxes[idx])

            return all_X, all_general_labels, all_gt_boxes
    
    def __len__(self):
        return self.X.shape[0]



