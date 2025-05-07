#!/usr/bin/env python
# -*- coding: utf-8 -*-
# commentary：Swintransformer-b-224，先拼接再进入。res特征融合模块：IAFF 细化：FM
#
# @Time    : 2024/5/19 12:08
# @Author  : Tyu
# @Site    :
# @File    : HaNet.py
# @Software: PyCharm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbone.swintransformer.swintransformer as st
import backbone.resnet.resnet as resnet
import backbone.Swin as swin
from modules import Decoder as DecoderBlock
from modules import norm_layer, ChannelCompression, upsample, PredictLayer


# 通道注意力模块
###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


# 空间注意力模块
###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


# 用于分心策略的CE模块
###################################################################
# ################## Context Exploration Block ####################
###################################################################
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce


class Pyramid_Positioning(nn.Module):
    def __init__(self, channel):
        super(Pyramid_Positioning, self).__init__()
        self.channel = channel
        self.channel_quarter = int(channel / 4)

        self.conv5 = nn.Sequential(nn.Conv2d(self.channel, self.channel_quarter, 3, 1, 1),
                                   nn.BatchNorm2d(self.channel_quarter), nn.ReLU(),
                                   nn.AdaptiveMaxPool2d((5, 5)))
        self.conv7 = nn.Sequential(nn.Conv2d(self.channel, self.channel_quarter, 3, 1, 1),
                                   nn.BatchNorm2d(self.channel_quarter), nn.ReLU(),
                                   nn.AdaptiveMaxPool2d((7, 7)))
        self.conv9 = nn.Sequential(nn.Conv2d(self.channel, self.channel_quarter, 3, 1, 1),
                                   nn.BatchNorm2d(self.channel_quarter), nn.ReLU(),
                                   nn.AdaptiveMaxPool2d((9, 9)))
        self.conv11 = nn.Sequential(nn.Conv2d(self.channel, self.channel_quarter, 3, 1, 1),
                                    nn.BatchNorm2d(self.channel_quarter), nn.ReLU(),
                                    nn.AdaptiveMaxPool2d((11, 11)))

        self.pm5 = Positioning(self.channel_quarter)
        self.pm7 = Positioning(self.channel_quarter)
        self.pm9 = Positioning(self.channel_quarter)
        self.pm11 = Positioning(self.channel_quarter)

        # self.up5 = nn.UpsamplingBilinear2d(size=(13, 13))
        # self.up7 = nn.UpsamplingBilinear2d(size=(13, 13))
        # self.up9 = nn.UpsamplingBilinear2d(size=(13, 13))
        # self.up11 = nn.UpsamplingBilinear2d(size=(13, 13))
        self.up5 = nn.UpsamplingBilinear2d(size=(14, 14))
        self.up7 = nn.UpsamplingBilinear2d(size=(14, 14))
        self.up9 = nn.UpsamplingBilinear2d(size=(14, 14))
        self.up11 = nn.UpsamplingBilinear2d(size=(14, 14))
        # self.up5 = nn.UpsamplingBilinear2d(size=(11, 11))
        # self.up7 = nn.UpsamplingBilinear2d(size=(11, 11))
        # self.up9 = nn.UpsamplingBilinear2d(size=(11, 11))
        # self.up11 = nn.UpsamplingBilinear2d(size=(11, 11))

        self.af = simam_module()
        self.fusion = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.BatchNorm2d(self.channel),
                                    nn.ReLU())
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        conv5 = self.conv5(x)
        conv7 = self.conv7(x)
        conv9 = self.conv9(x)
        conv11 = self.conv11(x)

        pm5 = self.pm5(conv5)[0]
        pm7 = self.pm7(conv7)[0]
        pm9 = self.pm9(conv9)[0]
        pm11 = self.pm11(conv11)[0]

        up5 = self.up5(pm5)
        up7 = self.up7(pm7)
        up9 = self.up9(pm9)
        up11 = self.up11(pm11)

        fusion = torch.cat([up5, up7, up9, up11], 1)
        fusion = self.fusion(fusion)
        fusion = fusion + x
        fusion = self.af(fusion)

        map = self.map(fusion)

        return fusion, map


# 用于初始预测的定位模块
###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        map = self.map(sab)

        return sab, map


class Positioning_hsi(nn.Module):
    def __init__(self, channel):
        super(Positioning_hsi, self).__init__()
        self.channel = channel
        # self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        # cab = self.cab(x)
        sab = self.sab(x)
        map = self.map(sab)

        return sab, map


# 深层特征融合


# 修正模块
###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        up = self.up(y)

        input_map = self.input_map(in_map)
        f_feature = x * input_map
        b_feature = x * (1 - input_map)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map


class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


###################################################################
# ##################### Context Enrichment Module #################
###################################################################
class Context_Enrichment_Module(nn.Module):
    def __init__(self, channel4, channel3, channel2, channel1):
        super(Context_Enrichment_Module, self).__init__()
        self.channel_gap = int(channel4 / 4)
        self.channel4 = channel4
        self.channel3 = channel3
        self.channel2 = channel2
        self.channel1 = channel1
        self.c4_w1 = nn.Parameter(torch.ones(1))
        self.c4_w2 = nn.Parameter(torch.ones(1))
        self.c4_w4 = nn.Parameter(torch.ones(1))
        self.c4_w8 = nn.Parameter(torch.ones(1))
        self.c3_w1 = nn.Parameter(torch.ones(1))
        self.c3_w2 = nn.Parameter(torch.ones(1))
        self.c3_w4 = nn.Parameter(torch.ones(1))
        self.c3_w8 = nn.Parameter(torch.ones(1))
        self.c2_w1 = nn.Parameter(torch.ones(1))
        self.c2_w2 = nn.Parameter(torch.ones(1))
        self.c2_w4 = nn.Parameter(torch.ones(1))
        self.c2_w8 = nn.Parameter(torch.ones(1))
        self.c1_w1 = nn.Parameter(torch.ones(1))
        self.c1_w2 = nn.Parameter(torch.ones(1))
        self.c1_w4 = nn.Parameter(torch.ones(1))
        self.c1_w8 = nn.Parameter(torch.ones(1))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                 nn.Conv2d(self.channel4, self.channel_gap, 1), nn.BatchNorm2d(self.channel_gap),
                                 nn.ReLU(),
                                 nn.Conv2d(self.channel_gap, self.channel4, 1), nn.BatchNorm2d(self.channel4),
                                 nn.ReLU())
        self.up43 = nn.Sequential(nn.Conv2d(self.channel4, self.channel3, 7, 1, 3), nn.BatchNorm2d(self.channel3),
                                  nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up32 = nn.Sequential(nn.Conv2d(self.channel3, self.channel2, 7, 1, 3), nn.BatchNorm2d(self.channel2),
                                  nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up21 = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3), nn.BatchNorm2d(self.channel1),
                                  nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.c4_1 = nn.Sequential(nn.Conv2d(self.channel4, self.channel4, 3, 1, 1, dilation=1),
                                  nn.BatchNorm2d(self.channel4), nn.ReLU())
        self.c3_1 = nn.Sequential(nn.Conv2d(self.channel3, self.channel3, 3, 1, 1, dilation=1),
                                  nn.BatchNorm2d(self.channel3), nn.ReLU())
        self.c2_1 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 1, dilation=1),
                                  nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.c1_1 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 1, 1, dilation=1),
                                  nn.BatchNorm2d(self.channel1), nn.ReLU())
        self.c4_2 = nn.Sequential(nn.Conv2d(self.channel4, self.channel4, 3, 1, 2, dilation=2),
                                  nn.BatchNorm2d(self.channel4), nn.ReLU())
        self.c3_2 = nn.Sequential(nn.Conv2d(self.channel3, self.channel3, 3, 1, 2, dilation=2),
                                  nn.BatchNorm2d(self.channel3), nn.ReLU())
        self.c2_2 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 2, dilation=2),
                                  nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.c1_2 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 1, 2, dilation=2),
                                  nn.BatchNorm2d(self.channel1), nn.ReLU())
        self.c4_4 = nn.Sequential(nn.Conv2d(self.channel4, self.channel4, 3, 1, 4, dilation=4),
                                  nn.BatchNorm2d(self.channel4), nn.ReLU())
        self.c3_4 = nn.Sequential(nn.Conv2d(self.channel3, self.channel3, 3, 1, 4, dilation=4),
                                  nn.BatchNorm2d(self.channel3), nn.ReLU())
        self.c2_4 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 4, dilation=4),
                                  nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.c1_4 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 1, 4, dilation=4),
                                  nn.BatchNorm2d(self.channel1), nn.ReLU())
        self.c4_8 = nn.Sequential(nn.Conv2d(self.channel4, self.channel4, 3, 1, 8, dilation=8),
                                  nn.BatchNorm2d(self.channel4), nn.ReLU())
        self.c3_8 = nn.Sequential(nn.Conv2d(self.channel3, self.channel3, 3, 1, 8, dilation=8),
                                  nn.BatchNorm2d(self.channel3), nn.ReLU())
        self.c2_8 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 8, dilation=8),
                                  nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.c1_8 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 1, 8, dilation=8),
                                  nn.BatchNorm2d(self.channel1), nn.ReLU())
        self.cf4 = simam_module()
        self.cf3 = simam_module()
        self.cf2 = simam_module()
        self.cf1 = simam_module()
        self.output4 = nn.Sequential(nn.Conv2d(self.channel4, self.channel4, 3, 1, 1), nn.BatchNorm2d(self.channel4),
                                     nn.ReLU())
        self.output3 = nn.Sequential(nn.Conv2d(self.channel3, self.channel3, 3, 1, 1), nn.BatchNorm2d(self.channel3),
                                     nn.ReLU())
        self.output2 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 1), nn.BatchNorm2d(self.channel2),
                                     nn.ReLU())
        self.output1 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 1, 1), nn.BatchNorm2d(self.channel1),
                                     nn.ReLU())

    def forward(self, feature4, feature3, feature2, feature1):
        c4_input = feature4 + self.gap(feature4)
        c4_1 = self.c4_1(c4_input)
        c4_2 = self.c4_2(c4_input)
        c4_4 = self.c4_4(c4_input)
        c4_8 = self.c4_8(c4_input)
        c4 = self.c4_w1 * c4_1 + self.c4_w2 * c4_2 + self.c4_w4 * c4_4 + self.c4_w8 * c4_8
        c4 = self.cf4(c4)
        output4 = self.output4(feature4 + c4)

        c3_input = feature3 + self.up43(c4)
        c3_1 = self.c3_1(c3_input)
        c3_2 = self.c3_2(c3_input)
        c3_4 = self.c3_4(c3_input)
        c3_8 = self.c3_8(c3_input)
        c3 = self.c3_w1 * c3_1 + self.c3_w2 * c3_2 + self.c3_w4 * c3_4 + self.c3_w8 * c3_8
        c3 = self.cf3(c3)
        output3 = self.output3(feature3 + c3)

        c2_input = feature2 + self.up32(c3)
        c2_1 = self.c2_1(c2_input)
        c2_2 = self.c2_2(c2_input)
        c2_4 = self.c2_4(c2_input)
        c2_8 = self.c2_8(c2_input)
        c2 = self.c2_w1 * c2_1 + self.c2_w2 * c2_2 + self.c2_w4 * c2_4 + self.c2_w8 * c2_8
        c2 = self.cf2(c2)
        output2 = self.output2(feature2 + c2)

        c1_input = feature1 + self.up21(c2)
        c1_1 = self.c1_1(c1_input)
        c1_2 = self.c1_2(c1_input)
        c1_4 = self.c1_4(c1_input)
        c1_8 = self.c1_8(c1_input)
        c1 = self.c1_w1 * c1_1 + self.c1_w2 * c1_2 + self.c1_w4 * c1_4 + self.c1_w8 * c1_8
        c1 = self.cf1(c1)
        output1 = self.output1(feature1 + c1)

        return output4, output3, output2, output1


###################################################################
# ########################## NETWORK ##############################
###################################################################
class HaNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(HaNet, self).__init__()
        # params

        # backbone
        # self.ST_hsi = st.SwinTransformerFeatureExtractor_hsi()
        # self.ST_rgb = st.SwinTransformerFeatureExtractor_rgb()
        # self.ST_joint = st.SwinTransformerFeatureExtractor_joint()
        # resnet50 = resnet.resnet50(backbone_path=None, pretrained=False)
        resnet50_rgb = resnet.resnet50(backbone_path='./backbone\\resnet\\resnet50-19c8e357.pth', pretrained=True)
        self.layer0_rgb_r = nn.Sequential(resnet50_rgb.conv1, resnet50_rgb.bn1, resnet50_rgb.relu)
        self.layer1_rgb_r = nn.Sequential(resnet50_rgb.maxpool, resnet50_rgb.layer1)
        self.layer2_rgb_r = resnet50_rgb.layer2
        self.layer3_rgb_r = resnet50_rgb.layer3
        self.layer4_rgb_r = resnet50_rgb.layer4

        resnet50_hsi = resnet.resnet50(backbone_path='./backbone\\resnet\\resnet50-19c8e357.pth', pretrained=True)
        resnet50_hsi.conv1 = nn.Conv2d(31, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer0_hsi_r = nn.Sequential(resnet50_hsi.conv1, resnet50_hsi.bn1, resnet50_hsi.relu)
        self.layer1_hsi_r = nn.Sequential(resnet50_hsi.maxpool, resnet50_hsi.layer1)
        self.layer2_hsi_r = resnet50_hsi.layer2
        self.layer3_hsi_r = resnet50_hsi.layer3
        self.layer4_hsi_r = resnet50_hsi.layer4

        self.swin = swin.Swintransformer(224)
        self.swin.load_state_dict(torch.load('./backbone/swin_large_patch4_window7_224_22k.pth')['model'], strict=False)

        # channel reduction
        self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.cr4_h = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3_h = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2_h = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1_h = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.cr4rgb_swin = nn.Sequential(nn.Conv2d(768, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())

        self.cem = Context_Enrichment_Module(512, 256, 128, 64)

        # positioning
        self.positioning_rgb = Positioning(512)
        self.positioning_hsi = Positioning_hsi(512)
        self.positioning_pyr = Pyramid_Positioning(512)

        # focus
        self.focus3_rgb = Focus(256, 512)
        self.focus2_rgb = Focus(128, 256)

        self.focus1 = Focus(64, 128)

        self.focus3_hsi = Focus(256, 512)
        self.focus2_hsi = Focus(128, 256)
        self.focus1_hsi = Focus(64, 128)

        self.decoder = DecoderBlock(64)

        self.compress2 = ChannelCompression(128, 64)
        self.compress3 = ChannelCompression(256, 64)

        self.map = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, 7, 1, 3))
        self.pre = nn.Conv2d(64, 1, 7, 1, 3)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, rgb, hsi):
        # rgb: [batch_size, channel=3, h, w]
        # hsi: [batch_size, channel=31, h, w]
        # St_hsi = self.ST_hsi(hsi)
        # St_rgb = self.ST_rgb(rgb)
        # St_joint = self.ST_joint(rgb, hsi)

        layer0_rgb_r = self.layer0_rgb_r(rgb)  # [-1, 64, h/2, w/2]
        layer1_rgb_r = self.layer1_rgb_r(layer0_rgb_r)  # [-1, 256, h/4, w/4]
        layer2_rgb_r = self.layer2_rgb_r(layer1_rgb_r)  # [-1, 512, h/8, w/8]
        layer3_rgb_r = self.layer3_rgb_r(layer2_rgb_r)  # [-1, 1024, h/16, w/16]
        layer4_rgb_r = self.layer4_rgb_r(layer3_rgb_r)  # [-1, 2048, h/32, w/32]

        y = F.interpolate(rgb, size=(224, 224), mode='bilinear', align_corners=True)
        s1, s2, s3, s4 = self.swin(y)

        layer0_hsi_r = self.layer0_hsi_r(hsi)  # [-1, 64, h/2, w/2]
        layer1_hsi_r = self.layer1_hsi_r(layer0_hsi_r)  # [-1, 256, h/4, w/4]
        layer2_hsi_r = self.layer2_hsi_r(layer1_hsi_r)  # [-1, 512, h/8, w/8]
        layer3_hsi_r = self.layer3_hsi_r(layer2_hsi_r)  # [-1, 1024, h/16, w/16]
        layer4_hsi_r = self.layer4_hsi_r(layer3_hsi_r)  # [-1, 2048, h/32, w/32]

        # channel reduction
        cr4_rgb = self.cr4(layer4_rgb_r)  # 512
        cr3_rgb = self.cr3(layer3_rgb_r)  # 256
        cr2_rgb = self.cr2(layer2_rgb_r)  # 128
        cr1_rgb = self.cr1(layer1_rgb_r)  # 64

        cr4_rgb_s = self.cr4rgb_swin(s4)

        cr4_hsi = self.cr4_h(layer4_hsi_r)
        cr3_hsi = self.cr3_h(layer3_hsi_r)
        cr2_hsi = self.cr2_h(layer2_hsi_r)
        cr1_hsi = self.cr1_h(layer1_hsi_r)

        # cr4_rgb_s,cr3_hsi,cr2_hsi,cr1_hsi = self.cem(cr4_rgb_s,cr3_rgb,cr2_rgb,cr1_rgb)

        # cr1_joint = self.AFF1(cr1_rgb, cr1_hsi)
        # cr2_joint = self.AFF2(cr2_rgb, cr2_hsi)
        # cr3_joint = self.AFF3(cr3_rgb, cr3_hsi)
        # cr4_joint = self.AFF4(cr4_rgb, cr4_hsi)

        # positioning
        # SFF = self.AF(St_rgb, St_hsi)
        positioning_rgb, predict4_rgb = self.positioning_rgb(cr4_rgb_s)

        # focus
        focus3, predict3_rgb = self.focus3_rgb(cr3_rgb, positioning_rgb, predict4_rgb)
        focus2, predict2_rgb = self.focus2_rgb(cr2_rgb, focus3, predict3_rgb)
        focus1, predict1_rgb = self.focus1(cr1_rgb, focus2, predict2_rgb)

        positioning_hsi, predict4_hsi = self.positioning_hsi(cr4_hsi)

        positioning_fused = positioning_rgb + positioning_hsi

        focus3_hsi, predict3_hsi = self.focus3_hsi(cr3_hsi, positioning_fused, predict4_rgb)

        focus3_fused = focus3 + focus3_hsi
        focus2_hsi, predict2_hsi = self.focus2_hsi(cr2_hsi, focus3_fused, predict3_hsi)

        focus2_fused = focus2 + focus2_hsi
        focus1_hsi, predict1_hsi = self.focus1_hsi(cr1_hsi, focus2_fused, predict2_hsi)

        fused_1 = focus1 + focus1_hsi
        fused_1 = self.decoder(fused_1, focus1)



        # 加一个内容增强模块、让RGB走一遍SwinTransformer-L的backbone
        # 对HSI进行降维

        # 这里改成加法操作
        # cr_focus = focus1 + focus1_hsi
        pre1 = self.map(fused_1)

        # rescale
        predict4_rgb = F.interpolate(predict4_rgb, size=rgb.size()[2:], mode='bilinear', align_corners=True)
        predict4_fused = F.interpolate(predict4_hsi, size=rgb.size()[2:], mode='bilinear', align_corners=True)
        predict3_rgb = F.interpolate(predict3_rgb, size=rgb.size()[2:], mode='bilinear', align_corners=True)
        predict3_fused = F.interpolate(predict3_hsi, size=rgb.size()[2:], mode='bilinear', align_corners=True)
        predict2_rgb = F.interpolate(predict2_rgb, size=rgb.size()[2:], mode='bilinear', align_corners=True)
        predict2_fused = F.interpolate(predict2_hsi, size=rgb.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(pre1, size=rgb.size()[2:], mode='bilinear', align_corners=True)
        if self.training:
            return predict4_rgb, predict4_fused, predict3_rgb, predict3_fused, predict2_rgb, predict2_fused, predict1

        return torch.sigmoid(predict4_rgb), torch.sigmoid(predict4_fused), torch.sigmoid(predict3_rgb), torch.sigmoid(
            predict3_fused), torch.sigmoid(predict2_rgb), torch.sigmoid(
            predict2_fused),torch.sigmoid(predict1)


if __name__ == "__main__":
    rgb = torch.randn((11, 3, 448, 448)).cuda()
    hsi = torch.randn((11, 31, 448, 448)).cuda()
    model = HaNet().cuda()
    l1, l2, l3, L4, L5, L6, L7 = model(rgb, hsi)
    print(l1, l2, l3)
