import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def get_data_table_DET0102(au_table_path='',
                            label_path='',
                            multi_angle_offset=0,
                            usecols=None):
    # read in label file
    v1_hi = pd.read_csv(label_path,header=9, usecols=['temporal_coordinates','spatial_coordinates','metadata'])
    v1_hi = v1_hi[(v1_hi['temporal_coordinates'] != '[]') & (v1_hi['spatial_coordinates'] == '[]')]
    tics = np.vstack([eval(e) for e in v1_hi['temporal_coordinates'].tolist()])
    v1_hi['start_s'] = tics[:,0] + multi_angle_offset
    v1_hi['end_s'] = tics[:,1] + multi_angle_offset

    # read in OpenFace predicted au table
    if usecols is None:
        usecols = []
        usecols.extend(['frame',' timestamp',' confidence',' success'])
        AU_i = [' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',' AU17_r',' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r']
        AU_p = [' AU01_c',' AU02_c',' AU04_c',' AU05_c',' AU06_c',' AU07_c',' AU09_c',' AU10_c',' AU12_c',' AU14_c',' AU15_c',' AU17_c',' AU20_c',' AU23_c',' AU25_c',' AU26_c',' AU28_c',' AU45_c']
        usecols.extend(AU_i)
        usecols.extend(AU_p)
    v1_hi_data = pd.read_csv(au_table_path, usecols=usecols)

    # get labels for each annotation
    fine_labels = []
    for ele in [eval(e) for e in v1_hi['metadata'].tolist()]:
        assert ele.get('1') is not None
        fine_labels.append(int(ele.get('1')))
    fine_labels = np.array(fine_labels)
    general_labels = (fine_labels > 3.5) * 1 # 0,1,2,3 are non-tic behaviors
    v1_hi['fine_labels'] = fine_labels
    v1_hi['general_labels'] = general_labels
  
    # get label for each frame
    fine_mat = np.zeros((len(v1_hi_data), 15))
    for r in v1_hi.iterrows():
        mask = (v1_hi_data[' timestamp'] >= r[1]['start_s']) & (v1_hi_data[' timestamp'] <= r[1]['end_s'])
        fine_mat[mask, r[1]['fine_labels']] = 1 # r[1]['fine_labels']
    v1_hi_data = pd.concat([v1_hi_data,pd.DataFrame(fine_mat)], axis=1)
    
    v1_hi_data['general_labels'] = (fine_mat[:,4:].sum(1) > 0) * 1

    # decide where to start and end
    START = v1_hi_data['general_labels'].tolist().index(1)
    END = len(v1_hi_data) - v1_hi_data['general_labels'].tolist()[::-1].index(1)

    # extend start and end
    START = max(START-100, 0)
    END = min(END+100, len(v1_hi_data))
    data_to_use = v1_hi_data[START:END]

    # a = data_to_use['frame'].values
    # assert np.unique(a[1:]-a[:-1]).shape[0] == 1

    return data_to_use


