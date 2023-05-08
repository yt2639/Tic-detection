import math
import torch
from torch import nn
import torch.nn.functional as F
from loss import *
from detection import Detection1D
from unet import *

class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=64):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv1d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv1d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv1d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv1d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs # C2是layer1出来的，最coarse，只downsample 4x
        
        # C5 (2,)
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P4_upsampled_x + P3_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P3_upsampled_x + P2_x
        P2_x = self.P2_2(P2_x)

        return [P2_x, P3_x, P4_x, P5_x] # f4, f8, f16, f32


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding=1"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


#### heads
class RegHead(nn.Module):
    def __init__(self,channel_in,channel_out,feature_channel=64):
        super(RegHead,self).__init__()

        self.conv1 = nn.Conv1d(channel_in, feature_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, padding=1)

        # self.conv3 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, padding=1)

        self.output = nn.Conv1d(feature_channel, channel_out, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        # out = self.conv2(out)
        # out = self.relu(out)
        out = self.output(out)

        return out#.view(out.shape[0], -1, 2).contiguous()

class RegHead_multiconv(nn.Module):
    def __init__(self,channel_in,channel_out,feature_channel=64):
        super(RegHead_multiconv,self).__init__()
        self.conv1 = nn.Conv1d(channel_in, feature_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, padding=1)
        self.output = nn.Conv1d(feature_channel, channel_out, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(x))
        out = self.relu(self.conv3(x))
        out = self.output(out)

        return out

class ClfHead(nn.Module):
    def __init__(self,channel_in,channel_out,feature_channel=64):
        super(ClfHead,self).__init__()

        self.conv1 = nn.Conv1d(channel_in, feature_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Conv1d(feature_channel, channel_out, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.output(out)
        return out

class ClfHead_multiconv(nn.Module):
    def __init__(self,channel_in,channel_out,feature_channel=256):
        super(ClfHead_multiconv,self).__init__()

        self.conv1 = nn.Conv1d(channel_in, feature_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, padding=1)
        self.output = nn.Conv1d(feature_channel, channel_out, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.output(out)

        return out

class ClfHead_fc(nn.Module):
    def __init__(self,channel_in,channel_out,feature_channel=64):
        super(ClfHead_fc,self).__init__()
        self.fc1 = nn.Linear(channel_in, feature_channel)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, padding=1)
        self.fc2 = nn.Linear(feature_channel, channel_out)
        # self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.conv2(out)
        # out = self.relu(out)
        out = self.fc2(out)

        return out


### resnet
class ResNet_Large512(nn.Module):
    def __init__(self,block,layers,input_channels=17):
        self.CHANNELS = [512, 256, 256, 256, 256]
        self.FPN_CHANNEL = 256
        self.inplanes = self.CHANNELS[0]
        
        super(ResNet_Large512, self).__init__()

        self.conv1_1 = nn.Conv1d(input_channels, self.CHANNELS[0], kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1_1 = nn.BatchNorm1d(self.CHANNELS[0])
        self.conv1_2 = nn.Conv1d(self.CHANNELS[0], self.CHANNELS[0], kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1_2 = nn.BatchNorm1d(self.CHANNELS[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, self.CHANNELS[1], layers[0], stride=1)
        self.layer2 = self._make_layer(block, self.CHANNELS[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.CHANNELS[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.CHANNELS[4], layers[3], stride=2)

        fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,self.layer2[layers[1] - 1].conv2.out_channels, 
                     self.layer3[layers[2] - 1].conv2.out_channels,self.layer4[layers[3] - 1].conv2.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3], feature_size=self.FPN_CHANNEL)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                # torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        dropout = True
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x.shape  [2, 32, 104]

        x1 = self.layer1(x) # x1.shape [2, 32, 104]
        x2 = self.layer2(x1)# x2.shape [2, 64, 52]
        x3 = self.layer3(x2)# x3.shape [2, 64, 26]
        x4 = self.layer4(x3)# x4.shape [2, 64, 13]

        features = self.fpn([x1,x2,x3,x4])

        if dropout:
            features[0] = torch._VF.feature_dropout(features[0], 0.4, train=self.training)
            features[1] = torch._VF.feature_dropout(features[1], 0.4, train=self.training)
            features[2] = torch._VF.feature_dropout(features[2], 0.4, train=self.training)
            features[3] = torch._VF.feature_dropout(features[3], 0.4, train=self.training)
        return features

# ResNet + UNet
class ResNet_UNet3p(nn.Module):
    def __init__(self,block,layers,input_channels=17, unet_nlayers=5):
        ## -------------Encoder--------------
        # if input_channels == 17:
        #     self.CHANNELS = [32,32,64,64,64]
        #     self.FPN_CHANNEL = 64
        #     self.inplanes = 32
        # else:
        #     self.CHANNELS = [64,64,64,128,128]
        #     self.FPN_CHANNEL = 128
        #     self.inplanes = 64

        # self.CHANNELS = [64+32, 128+32, 256+64, 256, 256]
        self.CHANNELS = [512, 256, 256, 256, 256]
        self.FPN_CHANNEL = 256
        self.inplanes = self.CHANNELS[0]
        
        self.DCT_H = [104,52,26,13]
        self.DCT_index = 0
        super(ResNet_UNet3p, self).__init__()

        self.unet_nlayers = unet_nlayers

        self.conv1_1 = nn.Conv1d(input_channels, self.CHANNELS[0], kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1_1 = nn.BatchNorm1d(self.CHANNELS[0])
        self.conv1_2 = nn.Conv1d(self.CHANNELS[0], self.CHANNELS[0], kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1_2 = nn.BatchNorm1d(self.CHANNELS[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # 这里又downsample了一下

        self.layer1 = self._make_layer(block, self.CHANNELS[1], layers[0], stride=1)
        self.layer2 = self._make_layer(block, self.CHANNELS[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.CHANNELS[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.CHANNELS[4], layers[3], stride=2)

        fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,self.layer2[layers[1] - 1].conv2.out_channels, 
                     self.layer3[layers[2] - 1].conv2.out_channels,self.layer4[layers[3] - 1].conv2.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3], feature_size=self.FPN_CHANNEL)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                # torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')

        ## -------------Decoder--------------
        if unet_nlayers == 5:
            self.unet = UNet_3Plus_DeepSup_5layers_for_detection(n_classes=1) # =1的话就是用sigmoid
        elif unet_nlayers == 3:
            self.unet = UNet_3Plus_DeepSup_3layers_for_detection(n_classes=1) # =1的话就是用sigmoid


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        
        if 'Fca' in str(block):
            dct_h = self.DCT_H[self.DCT_index]
            layers = [block(self.inplanes, planes, stride, downsample, dct_h=dct_h)]
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dct_h=dct_h))
            self.DCT_index += 1
        else:
            layers = [block(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        dropout = True
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x0 = self.relu(x) # [2, 32, 416]

        x = self.conv1_2(x0)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x.shape  [2, 32, 104]

        x1 = self.layer1(x) # x1.shape [2, 32, 104]
        x2 = self.layer2(x1)# x2.shape [2, 64, 52]
        x3 = self.layer3(x2)# x3.shape [2, 64, 26]
        x4 = self.layer4(x3)# x4.shape [2, 64, 13]

        features = self.fpn([x1,x2,x3,x4])

        if dropout:
            features[0] = torch._VF.feature_dropout(features[0], 0.4, train=self.training)
            features[1] = torch._VF.feature_dropout(features[1], 0.4, train=self.training)
            features[2] = torch._VF.feature_dropout(features[2], 0.4, train=self.training)
            features[3] = torch._VF.feature_dropout(features[3], 0.4, train=self.training)

        # UNet
        if self.unet_nlayers == 5:
            d1,d2,d3,d4,d5 = self.unet(x0,x1,x2,x3,x4) # d1...d5 shape (b,1,416)
            return features, d1,d2,d3,d4,d5
        elif self.unet_nlayers == 3:
            d1,d2,d3 = self.unet(x0,x1,x2) # d1...d5 shape (b,1,416)
            return features, d1,d2,d3



###### RetinaNet
class Det1D(nn.Module):
    def __init__(self, backbone_name='resnet18', device='cpu', anchor_type='old', clf_head='s_conv', num_classes=1,
                 thresh_low=0.1, thresh_high=0.5,
                 loss_type='DetLoss', is_focal=False, topk_match=8, iou_smooth=False, uni_mat=False, use_EIoU_loss=False, use_MAE_proba_loss=False,
                 neg_pos_ratio=2,
                 conf_threshold=0.01, nms_threshold=0.5, top_k=10, use_EIoU=False):
        super(Det1D, self).__init__()
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high

        self.is_focal = is_focal
        self.topk_match = topk_match
        self.iou_smooth = iou_smooth
        self.uni_mat = uni_mat
        self.use_EIoU_loss = use_EIoU_loss
        self.use_MAE_proba_loss = use_MAE_proba_loss
        self.use_EIoU = use_EIoU
        self.neg_pos_ratio = neg_pos_ratio
        self.num_classes = num_classes

        self.device = device
        prior = 0.01 # 1e-2

        input_channels = 17

        self.backbone = ResNet_Large512(BasicBlock, [2,2,2,2], input_channels=input_channels) # RF: 45,101,213,437

        CLF_HEAD = ClfHead
        if clf_head == 'fc':
            CLF_HEAD = ClfHead_fc
        elif clf_head == 'm_conv':
            CLF_HEAD = ClfHead_multiconv

        clf_layers, reg_layers = [], []
        out_channel = self.backbone.FPN_CHANNEL
        
        num_anchor = [6,6,6,12]
        levels = 4
        for level in range(levels):
            clf_layers.append(CLF_HEAD(out_channel, num_classes * num_anchor[level]))
            reg_layers.append(RegHead(out_channel, 2*num_anchor[level]))
        self.clf_layers = nn.Sequential(*clf_layers)
        self.reg_layers = nn.Sequential(*reg_layers)
        if is_focal:
            for clf, reg in zip(self.clf_layers, self.reg_layers):
                clf.output.weight.data.fill_(0)
                clf.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
                reg.output.weight.data.fill_(0)
                reg.output.bias.data.fill_(0)

        self.detect = Detection1D(conf_threshold=conf_threshold, nms_threshold=nms_threshold, top_k=top_k, use_EIoU=use_EIoU)
     
        self.loss_func = DetLoss()
        
    
    def forward(self,x,anchors,gt_boxes,neg_boxes=None,ignore_boxes=None):
        '''
        anchors: point form
        '''
        batch_size = x.size(0)
        features = self.backbone(x)
        # print(features.shape)
        clf_preds, reg_preds = [], []
        for feature,clf_layer,reg_layer in zip(features, self.clf_layers, self.reg_layers):
            clf_preds.append(clf_layer(feature).permute(0,2,1).contiguous().view(batch_size,-1))
            reg_preds.append(reg_layer(feature).permute(0,2,1).contiguous().view(batch_size,-1))
        clf_preds_all = torch.cat(clf_preds,1)
        reg_preds_all = torch.cat(reg_preds,1)
        clf_preds_all = clf_preds_all.view(batch_size, -1, self.num_classes)
        reg_preds_all = reg_preds_all.view(batch_size, -1, 2)
        # print(clf_preds_all.shape, reg_preds_all.shape)

        if self.training:
            clf_loss, reg_loss = self.loss_func(clf_preds_all, reg_preds_all, anchors, gt_boxes, neg_boxes=neg_boxes, ignore_boxes=ignore_boxes,
                                                device=self.device, neg_pos_ratio=self.neg_pos_ratio,
                                                thresh_low=self.thresh_low, thresh_high=self.thresh_high, 
                                                topk_match=self.topk_match, iou_smooth=self.iou_smooth,
                                                is_focal=self.is_focal, uni_mat=self.uni_mat, use_EIoU_loss=self.use_EIoU_loss, use_EIoU=self.use_EIoU, use_MAE_proba_loss=self.use_MAE_proba_loss)
            try:
                # for logging use
                self.eiou_loss = self.loss_func.EIoU_losses
                self.reg_loss = self.loss_func.reg_losses
            except:
                pass
            return clf_loss, reg_loss
        else:
            clf_proba_all = torch.sigmoid(clf_preds_all)
            self.clf_proba_all = clf_proba_all
            self.reg_preds_all = reg_preds_all
            detections = self.detect(clf_proba_all, reg_preds_all, anchors, self.device)
            self.ori_det = self.detect.ori_det
            return detections


class Det_Seg1D(nn.Module):
    def __init__(self, backbone_name='resnet18', det_seg_share_encoder=False, unet_nlayers=5, seg_ce_loss_type='focalBCE', multiple_scale_proba_vector=False,
                 device='cpu', anchor_type='old', clf_head='s_conv', num_classes=1,
                 thresh_low=0.1, thresh_high=0.5,
                 loss_type='FocalLoss', is_focal=False, topk_match=8, iou_smooth=False, uni_mat=False, use_EIoU_loss=False, use_MAE_proba_loss=False,
                 neg_pos_ratio=2,
                 conf_threshold=0.01, nms_threshold=0.5, top_k=10, use_EIoU=False,
                 input_channels=17):
        super(Det_Seg1D, self).__init__()
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high

        self.is_focal = is_focal
        self.topk_match = topk_match
        self.iou_smooth = iou_smooth
        self.uni_mat = uni_mat
        self.use_EIoU_loss = use_EIoU_loss
        self.use_MAE_proba_loss = use_MAE_proba_loss
        self.use_EIoU = use_EIoU
        self.neg_pos_ratio = neg_pos_ratio
        self.num_classes = num_classes
        self.det_seg_share_encoder = det_seg_share_encoder
        self.seg_ce_loss_type = seg_ce_loss_type
        self.multiple_scale_proba_vector = multiple_scale_proba_vector

        self.device = device
        prior = 0.01 # 1e-2

        CLF_HEAD = ClfHead
        # input_channels = 17

        if self.det_seg_share_encoder:
            self.backbone = ResNet_UNet3p(BasicBlock, [2,2,2,2], input_channels=input_channels, unet_nlayers=unet_nlayers) # 45,101,213,437
        else:
            self.backbone = ResNet_Large512(BasicBlock, [2,2,2,2], input_channels=input_channels) # 45,101,213,437
            if unet_nlayers == 3:
                self.unet = UNet_3Plus_DeepSup_3layers_Large512(in_channels=input_channels, n_classes=num_classes)

        clf_layers, reg_layers = [], []
        out_channel = self.backbone.FPN_CHANNEL
        
        num_anchor = [6,6,6,12]
        levels = 4
    
        for level in range(levels):
            clf_layers.append(CLF_HEAD(out_channel, num_classes * num_anchor[level]))
            reg_layers.append(RegHead(out_channel, 2*num_anchor[level]))
        self.clf_layers = nn.Sequential(*clf_layers)
        self.reg_layers = nn.Sequential(*reg_layers)
        if is_focal:
            for clf, reg in zip(self.clf_layers, self.reg_layers):
                clf.output.weight.data.fill_(0)
                clf.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
                reg.output.weight.data.fill_(0)
                reg.output.bias.data.fill_(0)

        self.detect = Detection1D(conf_threshold=conf_threshold, nms_threshold=nms_threshold, top_k=top_k, use_EIoU=use_EIoU)

        self.crop_seg_results_size = 25
        if multiple_scale_proba_vector:
            cat_channel = self.crop_seg_results_size * unet_nlayers + 1
            cat_channel_out = 256
        else:
            cat_channel = self.crop_seg_results_size + 1
            cat_channel_out = 128
        self.get_new_conf = nn.Sequential(
            nn.Linear(cat_channel, cat_channel_out),
            nn.BatchNorm1d(cat_channel_out),
            nn.ReLU(),
            nn.Linear(cat_channel_out, num_classes),
            # nn.ReLU(),
        )
    
    @torch.cuda.amp.autocast()
    def forward(self,x,anchors,seg_labels,gt_boxes,neg_boxes=None,ignore_boxes=None):

        batch_size = x.size(0)
        # features, d1,d2,d3,d4,d5 = self.backbone(x) 
        # features[0] (2,64,104), features[1] (2,64,52) ...
        # print(features)
        
        if self.det_seg_share_encoder:
            features_d = self.backbone(x) 
            features,d = features_d[0], features_d[1:]
            d1 = d[0]
        else:
            features = self.backbone(x) 
            d = self.unet(x) # d1...d5 shape (b,1,416)
            d1 = d[0]

        clf_preds, reg_preds = [], []
        for feature,clf_layer,reg_layer in zip(features, self.clf_layers, self.reg_layers):
            clf_preds.append(clf_layer(feature).permute(0,2,1).contiguous().view(batch_size,-1))
            reg_preds.append(reg_layer(feature).permute(0,2,1).contiguous().view(batch_size,-1))
        clf_preds_all = torch.cat(clf_preds,1)
        reg_preds_all = torch.cat(reg_preds,1)
        clf_preds_all = clf_preds_all.view(batch_size, -1, self.num_classes) # (2,1248,1)
        reg_preds_all = reg_preds_all.view(batch_size, -1, 2) # (2,1248,2)

        ### pool box on top of seg_results
        # print(anchors.shape, reg_preds_all.shape)
        # print(len(gt_boxes), len(neg_boxes), x.shape, seg_labels.shape)
        pred_boxes = BBoxTransform1D()(anchors, reg_preds_all) # (batch_size, num_anchors, 2) # (2,1248,2)
        pred_boxes = torch.clamp(pred_boxes, min=0., max=416.)
        pred_boxes = pred_boxes.detach().float()
        num_anchors = pred_boxes.shape[1]

        # roipooling
        pred_boxes = pred_boxes.reshape(-1,2)
        lengths = pred_boxes[:,1] - pred_boxes[:,0]
        assign_0_mask = lengths <= 0
        # create rois
        batch_index = torch.arange(batch_size, dtype=torch.float).repeat((num_anchors,1)).t().flatten().reshape(-1,1).to(d1.device)
        y1 = torch.zeros(batch_index.shape).to(d1.device)
        y2 = torch.ones(batch_index.shape).to(d1.device)
        # print(batch_index.shape, pred_boxes[:,0:1].shape,y1.shape, pred_boxes[1:2].shape,y2.shape)
        # print(batch_index.dtype, pred_boxes[:,0:1].dtype,y1.dtype, pred_boxes[1:2].dtype,y2.dtype)
        rois = torch.cat([batch_index, pred_boxes[:,0:1],y1, pred_boxes[:,1:2],y2], 1) # batch_index, x1,y1, x2,y2  y is height, x is width，so y1=0, y2=1
        rois = rois.type(d1.dtype)

        # d1.unsqueeze(-2) shape (b,1,1,416)
        all_new_conf = torchvision.ops.roi_pool(d1.unsqueeze(-2), rois, output_size=(1, self.crop_seg_results_size)).squeeze() # (bxanchor#, 25)
        all_new_conf[assign_0_mask] = 0.
        all_new_conf = all_new_conf.reshape(batch_size, num_anchors, self.crop_seg_results_size)
        all_new_conf = torch.cat([clf_preds_all, all_new_conf], dim=-1) # (b, anchor#, 26)
  
        all_new_conf_shape = all_new_conf.shape # (512, 1248, 26)
        all_new_conf = self.get_new_conf(all_new_conf.view(-1,all_new_conf_shape[-1])).reshape(all_new_conf_shape[:-1]).unsqueeze(-1)

        if self.training:
            return all_new_conf, reg_preds_all, d

        else:
            clf_proba_all = torch.sigmoid(all_new_conf)

            self.clf_proba_all = clf_proba_all
            self.reg_preds_all = reg_preds_all
            detections = self.detect(clf_proba_all, reg_preds_all, anchors, self.device)
            self.ori_det = self.detect.ori_det
            return detections, d1

