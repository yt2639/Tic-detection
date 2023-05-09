from sklearn.cluster import KMeans, kmeans_plusplus
import torch
import torch.nn as nn
import numpy as np


def kmeans_anchors(all_train_gt_boxes):
    train_gt_boxes_with_obj = torch.cat([e for e in all_train_gt_boxes if e.size()[0]>0])
    length = train_gt_boxes_with_obj[:,1] - train_gt_boxes_with_obj[:,0]
    tic_center = (train_gt_boxes_with_obj[:,1] + train_gt_boxes_with_obj[:,0])/2

    # center = np.array([15.5+i*32 for i in range(8)])
    # center = np.array([15.5+i*32 for i in range(13)])

    n_clusters = 10
    original_train_gt_boxes_with_obj = train_gt_boxes_with_obj.repeat(n_clusters,1,1)

    # # kmeans++ initialize 
    w = kmeans_plusplus(length.numpy().reshape(-1, 1),n_clusters,random_state=n_clusters)[0].reshape(-1,).astype(np.float)

    w_old = w.copy()

    counter=0
    while counter==0 or any(np.abs(w_old - w) > 0.01):
        counter+=1
        w_old = w.copy()

        new_tic = np.stack((tic_center.reshape(1,-1) - w.reshape(-1,1)/2, tic_center.reshape(1,-1) + w.reshape(-1,1)/2),2)
        inter = np.clip(np.minimum(original_train_gt_boxes_with_obj[:,:,1], new_tic[:,:,1]) - np.maximum(original_train_gt_boxes_with_obj[:,:,0], new_tic[:,:,0]), a_min=0,a_max=416)
        union = length.reshape(1,-1) + w.reshape(-1,1) - inter
        dist = 1-inter/union
        assign = dist.argmin(axis=0)

        for i in range(n_clusters):
            w[i] = length[assign == i].mean()

        how_many_nan = np.isnan(w).sum()
        if how_many_nan:
            print("There isn't {} clusters to assign, only {} clusters".format(n_clusters,n_clusters-how_many_nan))

    # # this proposal_boxes is in point_form!!
    # proposal_boxes = np.concatenate(np.stack((center.reshape(1,-1) - w.reshape(-1,1)/2, center.reshape(1,-1) + w.reshape(-1,1)/2),2))
    # proposal_boxes = torch.clamp(torch.from_numpy(proposal_boxes), 0, 415)
    # # this proposal_boxes is in center_form!!
    # # proposal_boxes = center_form(proposal_boxes)

    # print(proposal_boxes.shape)
    w = np.sort(w)
    print(w)

    #----------------------------------------------------------------------------- Multiscale的anchor
    w4 = np.sort(np.concatenate((w[:3],((np.append(0,w[:3]) + np.append(w[:3],0))/2)[:-1])))
    w8 = np.sort(np.concatenate((w[1:4],((np.append(0,w[1:4]) + np.append(w[1:4],0))/2)[:-1])))
    w16 = np.sort(np.concatenate((w[2:5],((np.append(0,w[2:5]) + np.append(w[2:5],0))/2)[:-1])))
    w32 = np.sort(np.concatenate((w[4:],((np.append(0,w[4:]) + np.append(w[4:],0))/2)[:-1])))
    w_all = [w4,w8,w16,w32]
    print(w4,'\n',w8,'\n',w16,'\n',w32,'\n')
    time_length = [104,52,26,13] # 每一个对应的grid的长度（downsample倍数） 4,8,16,32
    print(time_length)
    grid_center = [1.5, 3.5, 7.5, 15.5]
    downsample = [4,8,16,32]

    all_proposal_boxes = []
    for level in range(4):
        center = np.array([grid_center[level]+i*downsample[level] for i in range(time_length[level])])
        wi = w_all[level]
        proposal_boxes = np.concatenate(np.stack((center.reshape(1,-1) - wi.reshape(-1,1)/2, center.reshape(1,-1) + wi.reshape(-1,1)/2),2))
        # proposal_boxes = torch.clamp(torch.from_numpy(proposal_boxes), 0, 415)
        all_proposal_boxes.append(torch.from_numpy(proposal_boxes))

    all_proposal_boxes = torch.cat(all_proposal_boxes)
    print(all_proposal_boxes.shape)
    return all_proposal_boxes

def kmeans_anchors_no_repeat(all_train_gt_boxes):
    train_gt_boxes_with_obj = torch.cat([e for e in all_train_gt_boxes if e.size()[0]>0])
    length = train_gt_boxes_with_obj[:,1] - train_gt_boxes_with_obj[:,0]
    tic_center = (train_gt_boxes_with_obj[:,1] + train_gt_boxes_with_obj[:,0])/2

    # center = np.array([15.5+i*32 for i in range(8)])
    # center = np.array([15.5+i*32 for i in range(13)])

    n_clusters = 12
    original_train_gt_boxes_with_obj = train_gt_boxes_with_obj.repeat(n_clusters,1,1)

    # # kmeans++ initialize 
    w = kmeans_plusplus(length.numpy().reshape(-1, 1),n_clusters,random_state=n_clusters)[0].reshape(-1,).astype(np.float)

    w_old = w.copy()

    counter=0
    while counter==0 or any(np.abs(w_old - w) > 0.01):
        counter+=1
        w_old = w.copy()

        new_tic = np.stack((tic_center.reshape(1,-1) - w.reshape(-1,1)/2, tic_center.reshape(1,-1) + w.reshape(-1,1)/2),2)
        inter = np.clip(np.minimum(original_train_gt_boxes_with_obj[:,:,1], new_tic[:,:,1]) - np.maximum(original_train_gt_boxes_with_obj[:,:,0], new_tic[:,:,0]), a_min=0,a_max=416)
        union = length.reshape(1,-1) + w.reshape(-1,1) - inter
        dist = 1-inter/union
        assign = dist.argmin(axis=0)

        for i in range(n_clusters):
            w[i] = length[assign == i].mean()

        how_many_nan = np.isnan(w).sum()
        if how_many_nan:
            print("There isn't {} clusters to assign, only {} clusters".format(n_clusters,n_clusters-how_many_nan))

    # # this proposal_boxes is in point_form!!
    # proposal_boxes = np.concatenate(np.stack((center.reshape(1,-1) - w.reshape(-1,1)/2, center.reshape(1,-1) + w.reshape(-1,1)/2),2))
    # proposal_boxes = torch.clamp(torch.from_numpy(proposal_boxes), 0, 415)
    # # this proposal_boxes is in center_form!!
    # # proposal_boxes = center_form(proposal_boxes)

    # print(proposal_boxes.shape)
    w = np.sort(w)
    print(w)

    #----------------------------------------------------------------------------- Multiscale的anchor
    w4 = w[:3]
    w8 = w[3:6]
    w16 = w[6:9]
    w32 = w[9:12]
    
    w_all = [w4,w8,w16,w32]
    print(w4,'\n',w8,'\n',w16,'\n',w32,'\n')
    time_length = [104,52,26,13] # 每一个对应的grid的长度（downsample倍数） 4,8,16,32
    print(time_length)
    grid_center = [1.5, 3.5, 7.5, 15.5]
    downsample = [4,8,16,32]

    all_proposal_boxes = []
    for level in range(4):
        center = np.array([grid_center[level]+i*downsample[level] for i in range(time_length[level])])
        wi = w_all[level]
        proposal_boxes = np.concatenate(np.stack((center.reshape(1,-1) - wi.reshape(-1,1)/2, center.reshape(1,-1) + wi.reshape(-1,1)/2),2))
        # proposal_boxes = torch.clamp(torch.from_numpy(proposal_boxes), 0, 415)
        all_proposal_boxes.append(torch.from_numpy(proposal_boxes))

    all_proposal_boxes = torch.cat(all_proposal_boxes)
    print(all_proposal_boxes.shape)
    return all_proposal_boxes


# w = np.array([  9.08165741 , 19.34402275 , 30.08503914 , 43.18697739 , 58.96508789,
#                 80.84135437, 110.10414124 ,149.11904907 ,213.08781433,353.59042358])
def anchors_det0102(w = np.array([  9.08165741 , 19.34402275 , 30.08503914 , 43.18697739 , 58.96508789,
                                    80.84135437, 110.10414124 ,149.11904907 ,213.08781433,353.59042358])):
    w = np.sort(w)
    print(w)

    # Multiscale anchor
    w4 = np.sort(np.concatenate((w[:3],((np.append(0,w[:3]) + np.append(w[:3],0))/2)[:-1])))
    w8 = np.sort(np.concatenate((w[1:4],((np.append(0,w[1:4]) + np.append(w[1:4],0))/2)[:-1])))
    w16 = np.sort(np.concatenate((w[2:5],((np.append(0,w[2:5]) + np.append(w[2:5],0))/2)[:-1])))
    w32 = np.sort(np.concatenate((w[4:],((np.append(0,w[4:]) + np.append(w[4:],0))/2)[:-1])))
    w_all = [w4,w8,w16,w32]
    print(w4,'\n',w8,'\n',w16,'\n',w32,'\n')
    time_length = [104,52,26,13] # downsample 4,8,16,32
    print(time_length)
    grid_center = [1.5, 3.5, 7.5, 15.5]
    downsample = [4,8,16,32]

    all_proposal_boxes = []
    for level in range(4):
        center = np.array([grid_center[level]+i*downsample[level] for i in range(time_length[level])])
        wi = w_all[level]
        proposal_boxes = np.concatenate(np.stack((center.reshape(1,-1) - wi.reshape(-1,1)/2, center.reshape(1,-1) + wi.reshape(-1,1)/2),2))
        # proposal_boxes = torch.clamp(torch.from_numpy(proposal_boxes), 0, 415)
        all_proposal_boxes.append(torch.from_numpy(proposal_boxes))

    all_proposal_boxes = torch.cat(all_proposal_boxes)
    print(all_proposal_boxes.shape)
    return all_proposal_boxes
