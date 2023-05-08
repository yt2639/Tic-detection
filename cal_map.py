from iou import jaccard
from nms import nms
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import config

def combine_to_global(det_boxes, i_th=0, is_final=False, sample_stride=50, sample_length=416):
    if is_final:
        start = i_th
    else:
        start = i_th*sample_stride
        # end = (i_th+1)*sample_stride + sample_length
    det_boxes[:,0] = det_boxes[:,0] + start
    det_boxes[:,1] = det_boxes[:,1] + start
    return det_boxes

def cal_tp_fp_fn(test_video_names,
                detections,all_test_gt_boxes_names,GLOBAL_test_gt_boxes,
                sapa=None,experiment_name=None,write_out_results=False,
                CLASSES=None, CAL_MAP_TABLE=None,
                iou_threshold = 0.5):
    
    FP, TP, FN, MatchedGT = {},{},{},{}
    FP_names, TP_names, FN_names, MatchedGT_names = {},{},{},{}
    
    detections = detections.detach().cpu()

    # combine detections for one whole video
    all_det = {}
    for tvname in test_video_names:
        all_det[tvname] = []
        FP_names[tvname],TP_names[tvname], FN_names[tvname], MatchedGT_names[tvname] = [],[],[],[] 
        FP[tvname], TP[tvname], FN[tvname], MatchedGT[tvname] = [],[],[],[] 
    for i in range(detections.size(0)):
        det = detections[i,detections[i,:,-1] != -1,:]
        # apply margin constraint
        det = det[(det[:,0] >= config['eval_margin']-1) & (det[:,1] <= 416-config['eval_margin'])]
        if len(det) == 0: # no detection
            continue
        i_th, v_name = all_test_gt_boxes_names[i].dtype.names[0].split('-')
        is_final = False
        if i_th == 'final':
            i_th = CAL_MAP_TABLE[v_name].shape[0] - config['sample_length']
            is_final = True
        else:
            i_th = int(i_th)
        det = combine_to_global(det, i_th=i_th, is_final=is_final, sample_stride=config['sample_stride'], sample_length=config['sample_length'])
        all_det[v_name].append(det)

    # cal the tp and fp
    box_count,num_ignored = 0,0
    all_recall_precision = []
    for k,v in all_det.items():
        print(k)
        cur_v_gt_boxes = GLOBAL_test_gt_boxes[k]
        # cur_v = eval(k)
        cur_v = CAL_MAP_TABLE[k]

        pred_boxes_with_score = torch.cat(v) # (..., 3)
        print('before nms',pred_boxes_with_score.shape)
        pred_boxes,score = pred_boxes_with_score[:,:-1], pred_boxes_with_score[:,-1].reshape(-1,1)
        idx, count = nms(pred_boxes, score, 
                        nms_threshold=config['eval_nms_threshold'], top_k=config['eval_nms_topk'], use_EIoU=config['eval_nms_use_EIoU'])
        # print(count)
        box_count += count

        results = torch.cat((pred_boxes[idx], score[idx]),1) # (topk, 3) x,y,p
        print('after nms',results.shape)
        # sort
        results = results[torch.argsort(results[:,-1], descending=True)]

        # get TP Matched_GT FP FN
        already_detected_ids = []
        tp_id = torch.zeros(results.shape[0],1)
        n_iter = 0
        while (len(already_detected_ids)<len(cur_v_gt_boxes)) and (n_iter<len(results)-1):
            for n_iter,pred_box_with_score in enumerate(results):
                pred_box = pred_box_with_score[:-1]
                score = pred_box_with_score[-1]
                iou_v = jaccard(pred_box.reshape(1,2), cur_v_gt_boxes).squeeze()
                iou_max, iou_max_id = iou_v.max(dim=0)
                if (iou_max>=iou_threshold) and (iou_max_id not in already_detected_ids):
                    tp_id[n_iter] = 1
                    already_detected_ids.append(iou_max_id.item())
                    # TP
                    TP[k].append(pred_box_with_score)
                    st = int(np.floor(pred_box_with_score[0]))
                    ed = int(np.ceil(pred_box_with_score[1]))
                    time = cur_v.iloc[st:ed][' timestamp']
                    TP_names[k].append([k,*pred_box_with_score.tolist(),time.min(),time.max()])
        TP[k] = [torch.stack(TP[k])]
        # Matched GT
        MatchedGT[k].append(cur_v_gt_boxes[torch.tensor(already_detected_ids)].reshape(-1,2))
        for res in MatchedGT[k][0]:
            st = int(np.floor(res[0]))
            ed = int(np.ceil(res[1]))
            time = cur_v.iloc[st:ed][' timestamp']
            MatchedGT_names[k].append([k,*res.tolist(),66,time.min(),time.max()])
            
        # FN
        fn_idx = torch.tensor(list(set(range(len(cur_v_gt_boxes))).difference(set(already_detected_ids))))
        if fn_idx.numel() > 0:
            FN[k].append(cur_v_gt_boxes[fn_idx].reshape(-1,2))
            for res in FN[k][0]:
                st = int(np.floor(res[0]))
                ed = int(np.ceil(res[1]))
                time = cur_v.iloc[st:ed][' timestamp']
                FN_names[k].append([k,*res.tolist(),66,time.min(),time.max()])
         
        # FP
        fp_id = ~tp_id.bool() * 1
        FP[k].append(results[fp_id.squeeze() == 1])
        for res in FP[k][0]:
            st = int(np.floor(res[0]))
            ed = int(np.ceil(res[1]))
            time = cur_v.iloc[st:ed][' timestamp']
            FP_names[k].append([k,*res.tolist(),time.min(),time.max()])

        all_recall_precision.append(torch.cat((tp_id,fp_id,results[:,-1].reshape(-1,1)), dim=1))


    all_recall_precision = torch.cat(all_recall_precision)
    idx = all_recall_precision[:,-1].sort(descending=True)[1]
    all_recall_precision = all_recall_precision[idx]
    all_recall_precision = all_recall_precision.numpy()
    tp = np.cumsum(all_recall_precision[:,0])
    fp = np.cumsum(all_recall_precision[:,1])

    num_tp, num_fp = tp[-1], fp[-1]
    num_fn = np.sum([len(v[0]) for v in FN.values()])

    recall = tp / torch.cat(list(GLOBAL_test_gt_boxes.values())).shape[0]
    # precision = tp / (tp+fp)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # calculate ap
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    print('AP for Class Tic: {}'.format(ap,))

    # write out results summary
    if write_out_results:
        assert sapa is not None
        assert experiment_name is not None

        if not os.path.exists(sapa):
            os.makedirs(sapa)
        
        OutputFile = open(sapa+'RESULTS.txt', 'a')
        OutputFile.write('\n')
        OutputFile.write((60*'*')+'\n')
        OutputFile.write('Output results summary: \n')
        OutputFile.write(experiment_name + '\n')
        OutputFile.write((60*'*')+'\n')
        OutputFile.write('\n')

        OutputFile.write('Detection results: \n')
        OutputFile.write('\t TP: %d \n' % num_tp)
        OutputFile.write('\t FP: %d \n' % num_fp)
        OutputFile.write('\t FN: %d \n' % num_fn)
        OutputFile.write('\t Total # of detection boxes: %d \n' % box_count)
        OutputFile.write('\t Total # of tics: %d \n' % torch.cat(list(GLOBAL_test_gt_boxes.values())).shape[0])
        OutputFile.write('\t AP: %.2f \n' % float(ap*100))

        # OutputFile.write('\t Repeated detection on same tic (so ignored): %d \n' % num_ignored)
        # OutputFile.write('\t Recall: %.5f \n' % 0)
        # OutputFile.write('\t Avg FP per sample: %.5f \n' % 0)

        OutputFile.close()

        # write out csv
        GLOBAL_write_to_csv(FP_names, CLASSES, 'FP', sapa, CAL_MAP_TABLE)
        GLOBAL_write_to_csv(FN_names, CLASSES, 'FN', sapa, CAL_MAP_TABLE)
        GLOBAL_write_to_csv(TP_names, CLASSES, 'TP', sapa, CAL_MAP_TABLE)
        GLOBAL_write_to_csv(MatchedGT_names, CLASSES, 'MatchedGT', sapa, CAL_MAP_TABLE)

        # draw AP figure
        plt.plot(mrec, mpre, label='AP:{:.2f}'.format(float(ap*100)))
        plt.xlabel('recall',fontsize=14)
        plt.ylabel('precision',fontsize=14)
        plt.grid('on')
        plt.title(experiment_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(sapa+'AP_figure.png',bbox_inches=0,dpi=300)

    return mrec, mpre, ap, np.trapz(mpre,mrec), \
                TP, MatchedGT, FP, FN, num_tp, num_fp, num_fn, \
                TP_names, MatchedGT_names, FP_names, FN_names, \
                all_recall_precision


def GLOBAL_write_to_csv(names, CLASSES=None, TP_FP_FN=None, sapa=None, CAL_MAP_TABLE=None):
    assert TP_FP_FN is not None
    assert sapa is not None
    assert CLASSES is not None
    columns = ['video_name','start(frame,iloc)','end(frame,iloc)','probability','start(time)','end(time)','classes']

    rowlist = []
    for k,v in names.items():
        # cur_v = eval(k)
        cur_v = CAL_MAP_TABLE[k]
        for row in names[k]:
            s,e = row[-2], row[-1]
            cur_classes = cur_v[(float(s) <= cur_v[' timestamp']) & (cur_v[' timestamp'] <= float(e))][CLASSES].sum(0).values.nonzero()[0]
            classes = '_'.join([str(e) for e in cur_classes.tolist()])
            rowlist.append([*row,classes])
   
    data_csv = pd.DataFrame(rowlist, columns=columns)
    
    data_csv.to_csv(sapa+'{}s.csv'.format(TP_FP_FN), index=False)
    print('data saved to: ' + sapa+'{}s.csv'.format(TP_FP_FN))



