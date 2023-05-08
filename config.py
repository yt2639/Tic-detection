config = {}

config['use_data_parallel'] = [False,True][1]
if config['use_data_parallel']:
    config['batch_size'] = 512
    # config['batch_size'] = 864 # 6 gpus
else:
    config['batch_size'] = 128

config['num_anchors'] = 1248
config['input_channels'] = 17

config['sample_length'] = 416
config['sample_stride'] = 50

config['tic_class_offset'] = 4
config['eval_nms_topk'] = 2500
config['eval_nms_use_EIoU'] = [False,True][1]
if config['eval_nms_use_EIoU']:
    config['eval_nms_threshold'] = 0.05
else:
    config['eval_nms_threshold'] = 0.36
# margin setting
config['eval_margin'] = 32 # 50

