from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import pandas as pd
import time
from detect_steamlit import detect
import os
import sys
import argparse
from PIL import Image
import PIL
import base64
import torch
import cv2
import imageio
import sys
# set the matplotlib backend so figures can be saved in the background
import matplotlib
# import the necessary packages
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os
import pandas as pd
from imageio import imread
import keras
import cv2
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
from gtts import gTTS
import math
import gdown

# url = 'https://drive.google.com/uc?id=1kznuazx4fO8starp_JONzAeAtWjRnZ3A'
# output = 'D:/best(4).pt'
# gdown.download(url, output, quiet=True)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        numel = x.numel() / x.shape[0]
        return x.view(-1, int(numel))


def convNoutput(convs, input_size):  # predict output size after conv layers
    input_size = int(input_size)
    input_channels = convs[0][1].weight.shape[1]  # input channel
    output = torch.Tensor(1, input_channels, input_size, input_size)
    with torch.no_grad():
        for conv in convs:
            output = conv(output)
    return output.numel(), output.shape


class stn(nn.Module):
    def __init__(self, input_channels, input_size, params):
        super(stn, self).__init__()

        self.input_size = input_size

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(input_channels, params[0], kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(params[0], params[1], kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(params[1], params[2], kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        out_numel, out_size = convNoutput([self.conv1, self.conv2, self.conv3], input_size / 2)
        # set fc layer based on predicted size
        self.fc = nn.Sequential(
            View(),
            nn.Linear(out_numel, params[3]),
            nn.ReLU()
        )
        self.classifier = classifier = nn.Sequential(
            View(),
            nn.Linear(params[3], 6)  # affine transform has 6 parameters
        )
        # initialize stn parameters (affine transform)
        self.classifier[1].weight.data.fill_(0)
        self.classifier[1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def localization_network(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        theta = self.localization_network(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

class SillNet(nn.Module):
    def __init__(self, nc, input_size, class_train, class_test, extract_chn=None, classify_chn=None, param1=None, param2=None, param3=None, param4=None, param_mask=None):
        super(SillNet, self).__init__()

        self.extract_chn = extract_chn
        self.classify_chn = classify_chn
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4
        self.input_size = input_size
        self.nc = nc
        self.class_train = class_train
        self.class_test = class_test

        # extracter
        self.ex_pd1 = nn.ReplicationPad2d(2)
        self.ex1 = nn.Conv2d(nc, self.extract_chn[0], 5, 1)  # inchn, outchn, kernel, stride, padding, dilation, groups
        self.ex_bn1 = nn.InstanceNorm2d(self.extract_chn[0])

        self.ex_pd2 = nn.ReplicationPad2d(2)
        self.ex2 = nn.Conv2d(self.extract_chn[0], self.extract_chn[1], 5, 1)  # 1/1
        self.ex_bn2 = nn.InstanceNorm2d(self.extract_chn[1])

        self.ex_pd3 = nn.ReplicationPad2d(1)
        self.ex3 = nn.Conv2d(self.extract_chn[1], self.extract_chn[2], 3, 1)  # 1/1
        self.ex_bn3 = nn.InstanceNorm2d(self.extract_chn[2])

        self.ex_pd4 = nn.ReplicationPad2d(1)
        self.ex4 = nn.Conv2d(self.extract_chn[2], self.extract_chn[3], 3, 1)  # 1/1
        self.ex_bn4 = nn.InstanceNorm2d(self.extract_chn[3])

        self.ex_pd5 = nn.ReplicationPad2d(1)
        self.ex5 = nn.Conv2d(self.extract_chn[3], self.extract_chn[4], 3, 1)  # 1/1
        self.ex_bn5 = nn.InstanceNorm2d(self.extract_chn[4])

        self.ex_pd6 = nn.ReplicationPad2d(1)
        self.ex6 = nn.Conv2d(self.extract_chn[4], self.extract_chn[5], 3, 1)  # 1/1
        self.ex_bn6 = nn.InstanceNorm2d(self.extract_chn[5])

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # decoder
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.de_pd1 = nn.ReplicationPad2d(1)
        self.de1 = nn.Conv2d(int(self.extract_chn[5] / 2), self.extract_chn[4], 3, 1)
        self.de_bn1 = nn.InstanceNorm2d(self.extract_chn[4], 1.e-3)

        self.de_pd2 = nn.ReplicationPad2d(1)
        self.de2 = nn.Conv2d(self.extract_chn[4], self.extract_chn[3], 3, 1)
        self.de_bn2 = nn.InstanceNorm2d(self.extract_chn[3], 1.e-3)

        self.de_pd3 = nn.ReplicationPad2d(1)
        self.de3 = nn.Conv2d(self.extract_chn[3], self.extract_chn[2], 3, 1)
        self.de_bn3 = nn.InstanceNorm2d(self.extract_chn[2], 1.e-3)

        self.de_pd4 = nn.ReplicationPad2d(1)
        self.de4 = nn.Conv2d(self.extract_chn[2], self.extract_chn[1], 3, 1)
        self.de_bn4 = nn.InstanceNorm2d(self.extract_chn[1], 1.e-3)

        self.de_pd5 = nn.ReplicationPad2d(1)
        self.de5 = nn.Conv2d(self.extract_chn[1], nc, 3, 1)

        # warping
        if param1 is not None:
            self.stn1 = stn(nc, self.input_size, param1)
        if param2 is not None:
            self.stn2 = stn(self.extract_chn[1], self.input_size, param2)
        if param3 is not None:
            self.stn3 = stn(self.extract_chn[3], self.input_size, param3)
        if param4 is not None:
            self.stn4 = stn(int(self.extract_chn[5] / 2), self.input_size, param4)

        # classifier 1
        # inchn, outchn, kernel, stride, padding, dilation, groups
        self.cls1 = nn.Conv2d(int(self.extract_chn[5]), self.classify_chn[0], 5, 1, 2)
        self.cls_bn1 = nn.BatchNorm2d(self.classify_chn[0])

        self.cls2 = nn.Conv2d(self.classify_chn[0], self.classify_chn[1], 5, 1, 2)  # 1/2
        self.cls_bn2 = nn.BatchNorm2d(self.classify_chn[1])

        self.cls3 = nn.Conv2d(self.classify_chn[1], self.classify_chn[2], 5, 1, 2)  # 1/4
        self.cls_bn3 = nn.BatchNorm2d(self.classify_chn[2])

        self.cls4 = nn.Conv2d(self.classify_chn[2], self.classify_chn[3], 3, 1, 1)  # 1/4
        self.cls_bn4 = nn.BatchNorm2d(self.classify_chn[3])

        self.cls5 = nn.Conv2d(self.classify_chn[3], self.classify_chn[4], 3, 1, 1)  # 1/8
        self.cls_bn5 = nn.BatchNorm2d(self.classify_chn[4])

        self.cls6 = nn.Conv2d(self.classify_chn[4], self.classify_chn[5], 3, 1, 1)  # 1/8
        self.cls_bn6 = nn.BatchNorm2d(self.classify_chn[5])

        self.fc1 = nn.Linear(int(self.input_size / 8 * self.input_size / 8) * self.classify_chn[5], self.class_train)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        # classifier 2
        # inchn, outchn, kernel, stride, padding, dilation, groups
        self.cls21 = nn.Conv2d(int(self.extract_chn[5]), self.classify_chn[0], 5, 1, 2)
        self.cls2_bn1 = nn.BatchNorm2d(self.classify_chn[0])

        self.cls22 = nn.Conv2d(self.classify_chn[0], self.classify_chn[1], 5, 1, 2)  # 1/2
        self.cls2_bn2 = nn.BatchNorm2d(self.classify_chn[1])

        self.cls23 = nn.Conv2d(self.classify_chn[1], self.classify_chn[2], 5, 1, 2)  # 1/4
        self.cls2_bn3 = nn.BatchNorm2d(self.classify_chn[2])

        self.cls24 = nn.Conv2d(self.classify_chn[2], self.classify_chn[3], 3, 1, 1)  # 1/4
        self.cls2_bn4 = nn.BatchNorm2d(self.classify_chn[3])

        self.cls25 = nn.Conv2d(self.classify_chn[3], self.classify_chn[4], 3, 1, 1)  # 1/8
        self.cls2_bn5 = nn.BatchNorm2d(self.classify_chn[4])

        self.cls26 = nn.Conv2d(self.classify_chn[4], self.classify_chn[5], 3, 1, 1)  # 1/8
        self.cls2_bn6 = nn.BatchNorm2d(self.classify_chn[5])

        self.fc2 = nn.Linear(int(self.input_size / 8 * self.input_size / 8) * self.classify_chn[5], self.class_test)

    def extract(self, x, is_warping):
        if is_warping and self.param1 is not None:
            x = self.stn1(x)
        h1 = self.leakyrelu(self.ex_bn1(self.ex1(self.ex_pd1(x))))
        h2 = self.leakyrelu(self.ex_bn2(self.ex2(self.ex_pd2(h1))))

        if is_warping and self.param2 is not None:
            h2 = self.stn2(h2)
        h3 = self.leakyrelu(self.ex_bn3(self.ex3(self.ex_pd3(h2))))
        h4 = self.leakyrelu(self.ex_bn4(self.ex4(self.ex_pd4(h3))))

        if is_warping and self.param3 is not None:
            h4 = self.stn3(h4)
        h5 = self.leakyrelu(self.ex_bn5(self.ex5(self.ex_pd5(h4))))
        h6 = self.sigmoid(self.ex_bn6(self.ex6(self.ex_pd6(h5))))

        feat_sem, feat_illu = torch.chunk(h6, 2, 1)
        feat_sem_nowarp = feat_sem

        if is_warping and self.param4 is not None:
            feat_sem = self.stn4(feat_sem)

        return feat_sem, feat_illu, feat_sem_nowarp

    def decode(self, x):
        h1 = self.leakyrelu(self.de_bn1(self.de1(self.de_pd1(x))))
        h2 = self.leakyrelu(self.de_bn2(self.de2(self.de_pd2(h1))))
        h3 = self.leakyrelu(self.de_bn3(self.de3(self.de_pd3(h2))))
        h4 = self.leakyrelu(self.de_bn4(self.de4(self.de_pd4(h3))))
        out = self.sigmoid(self.de5(self.de_pd5(h4)))
        return out

    def classify(self, x):
        h1 = self.pool2(self.leakyrelu(self.cls_bn1(self.cls1(x))))
        h2 = self.leakyrelu(self.cls_bn2(self.cls2(h1)))
        h3 = self.pool2(self.leakyrelu(self.cls_bn3(self.cls3(h2))))
        h4 = self.leakyrelu(self.cls_bn4(self.cls4(h3)))
        h5 = self.pool2(self.leakyrelu(self.cls_bn5(self.cls5(h4))))
        h6 = self.leakyrelu(self.cls_bn6(self.cls6(h5)))
        h7 = h6.view(-1, int(self.input_size / 8 * self.input_size / 8 * self.classify_chn[5]))
        out = self.fc1(h7)
        return out

    def classify2(self, x):
        h1 = self.pool2(self.leakyrelu(self.cls2_bn1(self.cls21(x))))
        h2 = self.leakyrelu(self.cls2_bn2(self.cls22(h1)))
        h3 = self.pool2(self.leakyrelu(self.cls2_bn3(self.cls23(h2))))
        h4 = self.leakyrelu(self.cls2_bn4(self.cls24(h3)))
        h5 = self.pool2(self.leakyrelu(self.cls2_bn5(self.cls25(h4))))
        h6 = self.leakyrelu(self.cls2_bn6(self.cls26(h5)))
        h7 = h6.view(-1, int(self.input_size / 8 * self.input_size / 8 * self.classify_chn[5]))
        out = self.fc2(h7)
        return out

    def init_params(self, net):
        print('Loading the model from the file...')
        net_dict = self.state_dict()
        if isinstance(net, dict):
            pre_dict = net
        else:
            pre_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pre_dict = {k: v for k, v in pre_dict.items() if (k in net_dict)}
        net_dict.update(pre_dict)
        # 3. load the new state dict
        self.load_state_dict(net_dict)


class SillNet_gtsrb(nn.Module):
    def __init__(self, nc, input_size, class_train, class_test, extract_chn=None, classify_chn=None, param1=None, param2=None, param3=None, param4=None, param_mask=None):
        super(SillNet_gtsrb, self).__init__()

        self.extract_chn = extract_chn
        self.classify_chn = classify_chn
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4
        self.input_size = input_size
        self.nc = nc
        self.class_train = class_train
        self.class_test = class_test

        # extracter
        self.ex_pd1 = nn.ReplicationPad2d(2)
        self.ex1 = nn.Conv2d(nc, self.extract_chn[0], 5, 1)  # inchn, outchn, kernel, stride, padding, dilation, groups
        self.ex_bn1 = nn.InstanceNorm2d(self.extract_chn[0])

        self.ex_pd2 = nn.ReplicationPad2d(2)
        self.ex2 = nn.Conv2d(self.extract_chn[0], self.extract_chn[1], 5, 1)  # 1/1
        self.ex_bn2 = nn.InstanceNorm2d(self.extract_chn[1])

        self.ex_pd3 = nn.ReplicationPad2d(1)
        self.ex3 = nn.Conv2d(self.extract_chn[1], self.extract_chn[2], 3, 1)  # 1/1
        self.ex_bn3 = nn.InstanceNorm2d(self.extract_chn[2])

        self.ex_pd4 = nn.ReplicationPad2d(1)
        self.ex4 = nn.Conv2d(self.extract_chn[2], self.extract_chn[3], 3, 1)  # 1/1
        self.ex_bn4 = nn.InstanceNorm2d(self.extract_chn[3])

        self.ex_pd5 = nn.ReplicationPad2d(1)
        self.ex5 = nn.Conv2d(self.extract_chn[3], self.extract_chn[4], 3, 1)  # 1/1
        self.ex_bn5 = nn.InstanceNorm2d(self.extract_chn[4])

        self.ex_pd6 = nn.ReplicationPad2d(1)
        self.ex6 = nn.Conv2d(self.extract_chn[4], self.extract_chn[5], 3, 1)  # 1/1
        self.ex_bn6 = nn.InstanceNorm2d(self.extract_chn[5])

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # decoder
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.de_pd1 = nn.ReplicationPad2d(1)
        self.de1 = nn.Conv2d(int(self.extract_chn[5] / 2), self.extract_chn[4], 3, 1)
        self.de_bn1 = nn.InstanceNorm2d(self.extract_chn[4], 1.e-3)

        self.de_pd2 = nn.ReplicationPad2d(1)
        self.de2 = nn.Conv2d(self.extract_chn[4], self.extract_chn[3], 3, 1)
        self.de_bn2 = nn.InstanceNorm2d(self.extract_chn[3], 1.e-3)

        self.de_pd3 = nn.ReplicationPad2d(1)
        self.de3 = nn.Conv2d(self.extract_chn[3], self.extract_chn[2], 3, 1)
        self.de_bn3 = nn.InstanceNorm2d(self.extract_chn[2], 1.e-3)

        self.de_pd4 = nn.ReplicationPad2d(1)
        self.de4 = nn.Conv2d(self.extract_chn[2], self.extract_chn[1], 3, 1)
        self.de_bn4 = nn.InstanceNorm2d(self.extract_chn[1], 1.e-3)

        self.de_pd5 = nn.ReplicationPad2d(1)
        self.de5 = nn.Conv2d(self.extract_chn[1], nc, 3, 1)

        # warping
        if param1 is not None:
            self.stn1 = stn(nc, self.input_size, param1)
        if param2 is not None:
            self.stn2 = stn(self.extract_chn[1], self.input_size, param2)
        if param3 is not None:
            self.stn3 = stn(self.extract_chn[3], self.input_size, param3)
        if param4 is not None:
            self.stn4 = stn(int(self.extract_chn[5] / 2), self.input_size, param4)
            #self.rotate = rotate(int(self.extract_chn[5]/2), self.input_size, param4)

        # classifier 1
        # inchn, outchn, kernel, stride, padding, dilation, groups
        self.cls1 = nn.Conv2d(int(0.5 * self.extract_chn[5]), self.classify_chn[0], 5, 1, 2)
        self.cls_bn1 = nn.BatchNorm2d(self.classify_chn[0])

        self.cls2 = nn.Conv2d(self.classify_chn[0], self.classify_chn[1], 5, 1, 2)  # 1/2
        self.cls_bn2 = nn.BatchNorm2d(self.classify_chn[1])

        self.cls3 = nn.Conv2d(self.classify_chn[1], self.classify_chn[2], 5, 1, 2)  # 1/4
        self.cls_bn3 = nn.BatchNorm2d(self.classify_chn[2])

        self.cls4 = nn.Conv2d(self.classify_chn[2], self.classify_chn[3], 3, 1, 1)  # 1/4
        self.cls_bn4 = nn.BatchNorm2d(self.classify_chn[3])

        self.cls5 = nn.Conv2d(self.classify_chn[3], self.classify_chn[4], 3, 1, 1)  # 1/8
        self.cls_bn5 = nn.BatchNorm2d(self.classify_chn[4])

        self.cls6 = nn.Conv2d(self.classify_chn[4], self.classify_chn[5], 3, 1, 1)  # 1/8
        self.cls_bn6 = nn.BatchNorm2d(self.classify_chn[5])

        self.fc1 = nn.Linear(int(self.input_size / 8 * self.input_size / 8) * self.classify_chn[6], self.class_train)
        #self.glbpool = nn.MaxPool2d(kernel_size=int(self.input_size/8), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        # classifier 2
        # inchn, outchn, kernel, stride, padding, dilation, groups
        self.cls21 = nn.Conv2d(int(0.5 * self.extract_chn[5]), self.classify_chn[0], 5, 1, 2)
        self.cls2_bn1 = nn.BatchNorm2d(self.classify_chn[0])

        self.cls22 = nn.Conv2d(self.classify_chn[0], self.classify_chn[1], 5, 1, 2)  # 1/2
        self.cls2_bn2 = nn.BatchNorm2d(self.classify_chn[1])

        self.cls23 = nn.Conv2d(self.classify_chn[1], self.classify_chn[2], 5, 1, 2)  # 1/4
        self.cls2_bn3 = nn.BatchNorm2d(self.classify_chn[2])

        self.cls24 = nn.Conv2d(self.classify_chn[2], self.classify_chn[3], 3, 1, 1)  # 1/4
        self.cls2_bn4 = nn.BatchNorm2d(self.classify_chn[3])

        self.cls25 = nn.Conv2d(self.classify_chn[3], self.classify_chn[4], 3, 1, 1)  # 1/8
        self.cls2_bn5 = nn.BatchNorm2d(self.classify_chn[4])

        self.cls26 = nn.Conv2d(self.classify_chn[4], self.classify_chn[5], 3, 1, 1)  # 1/8
        self.cls2_bn6 = nn.BatchNorm2d(self.classify_chn[5])

        self.fc2 = nn.Linear(int(self.input_size / 8 * self.input_size / 8) * self.classify_chn[6], self.class_test)

    def extract(self, x, is_warping):
        if is_warping and self.param1 is not None:
            x = self.stn1(x)
        h1 = self.leakyrelu(self.ex_bn1(self.ex1(self.ex_pd1(x))))
        h2 = self.leakyrelu(self.ex_bn2(self.ex2(self.ex_pd2(h1))))

        if is_warping and self.param2 is not None:
            h2 = self.stn2(h2)
        h3 = self.leakyrelu(self.ex_bn3(self.ex3(self.ex_pd3(h2))))
        h4 = self.leakyrelu(self.ex_bn4(self.ex4(self.ex_pd4(h3))))

        if is_warping and self.param3 is not None:
            h4 = self.stn3(h4)
        h5 = self.leakyrelu(self.ex_bn5(self.ex5(self.ex_pd5(h4))))
        h6 = self.sigmoid(self.ex_bn6(self.ex6(self.ex_pd6(h5))))
        h6_nowarp = h6
        f_nowarp, noise = torch.chunk(h6_nowarp, 2, 1)

        feature, _ = torch.chunk(h6, 2, 1)

        if is_warping and self.param4 is not None:
            #feature = self.rotate(feature)
            feature = self.stn4(feature)

        return feature, noise, f_nowarp

    def decode(self, x):
        h1 = self.leakyrelu(self.de_bn1(self.de1(self.de_pd1(x))))
        h2 = self.leakyrelu(self.de_bn2(self.de2(self.de_pd2(h1))))
        h3 = self.leakyrelu(self.de_bn3(self.de3(self.de_pd3(h2))))
        h4 = self.leakyrelu(self.de_bn4(self.de4(self.de_pd4(h3))))
        out = self.sigmoid(self.de5(self.de_pd5(h4)))
        return out

    def classify(self, x):
        h1 = self.pool2(self.leakyrelu(self.cls_bn1(self.cls1(x))))
        h2 = self.leakyrelu(self.cls_bn2(self.cls2(h1)))
        h3 = self.pool2(self.leakyrelu(self.cls_bn3(self.cls3(h2))))
        h4 = self.leakyrelu(self.cls_bn4(self.cls4(h3)))
        h5 = self.pool2(self.leakyrelu(self.cls_bn5(self.cls5(h4))))
        h6 = self.leakyrelu(self.cls_bn6(self.cls6(h5)))
        h7 = h6.view(-1, int(self.input_size / 8 * self.input_size / 8 * self.classify_chn[6]))
        out = self.fc1(h7)
        #out = self.dropout(out)
        return out

    def classify2(self, x):
        h1 = self.pool2(self.leakyrelu(self.cls2_bn1(self.cls21(x))))
        h2 = self.leakyrelu(self.cls2_bn2(self.cls22(h1)))
        h3 = self.pool2(self.leakyrelu(self.cls2_bn3(self.cls23(h2))))
        h4 = self.leakyrelu(self.cls2_bn4(self.cls24(h3)))
        h5 = self.pool2(self.leakyrelu(self.cls2_bn5(self.cls25(h4))))
        h6 = self.leakyrelu(self.cls2_bn6(self.cls26(h5)))
        h7 = h6.view(-1, int(self.input_size / 8 * self.input_size / 8 * self.classify_chn[6]))
        out = self.fc2(h7)
        #out = self.dropout(out)
        return out

    def init_params(self, net):
        print('Loading the model from the file...')
        net_dict = self.state_dict()
        if isinstance(net, dict):
            pre_dict = net
        else:
            pre_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pre_dict = {k: v for k, v in pre_dict.items() if (k in net_dict)}  # for fs net
        net_dict.update(pre_dict)
        # 3. load the new state dict
        self.load_state_dict(net_dict)


def transform(img, img_size):
    img = img.astype(np.float64)
    #img -= self.mean
    if img_size is not None:
        img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
    # Resize scales images from 0 to 255, thus we need
    # to divide by 255.0
    img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.reshape(img, (1, 3, 64, 64))
    img = torch.from_numpy(img).float().to(torch.device('cpu'))
    return img


def predict_sign_class(molel, img):
    class_gt = {0: 'Speed limit (20 km/h)',
                1: 'Speed limit (30 km/h)',
                2: 'Speed limit (50 km/h)',
                3: 'Speed limit (60 km/h)',
                4: 'Speed limit (70km/h)',
                5: 'Speed limit (80km/h)',
                6: 'End of speed limit (80 km/h)',
                7: 'Speed limit (100 km/h)',
                8: 'Speed limit (120 km/h)',
                9: 'No passing',
                10: 'No passing veh over 3.5 tons',
                11: 'Right-of-way at intersection',
                12: 'Priority road',
                13: 'Yield',
                14: 'Stop',
                15: 'No vehicles',
                16: 'Veh > 3.5 tons prohibited',
                17: 'No entry',
                18: 'General caution',
                19: 'Dangerous curve left',
                20: 'Dangerous curve right',
                21: 'Double curve',
                22: 'Bumpy road',
                23: 'Slippery road',
                24: 'Road narrows on the right',
                25: 'Road work',
                26: 'Traffic signals',
                27: 'Pedestrians',
                28: 'Children crossing',
                29: 'Bicycles crossing',
                30: 'Beware of ice/snow',
                31: 'Wild animals crossing',
                32: 'End speed + passing limits',
                33: 'Turn right ahead',
                34: 'Turn left ahead',
                35: 'Ahead only',
                36: 'Go straight or right',
                37: 'Go straight or left',
                38: 'Keep right',
                39: 'Keep left',
                40: 'Roundabout mandatory',
                41: 'End of no passing',
                42: 'End no passing veh > 3.5 tons'}

    if isinstance(img, str):
        img = m.imread(img)
    # plt.imshow(img)
    img = np.array(img, dtype=np.uint8)
    img = transform(img, (64, 64))
    feat_sem, feat_illu, feat_nowarp = molel.extract(img, is_warping=True)
    feature_exc = torch.cat((feat_sem, feat_illu), 1)
    # class for dataset GTSRB
    prob, pred_sup = torch.max(molel.classify(feature_exc), 1)
    result = class_gt[pred_sup.tolist()[0]]
    # print(pred_sup)
    return result


def load_sillnet(weights):
    model_sillnet = SillNet(nc=3, input_size=64, class_train=43, class_test=43, extract_chn=[100, 150, 200, 150, 100, 6], classify_chn=[
                            100, 150, 200, 250, 300, 100], param1=None, param2=None, param3=None, param4=[150, 150, 150, 150])
    model_sillnet.init_params(torch.load(weights, map_location=torch.device('cpu')))
    return model_sillnet


def make_prediction(model_1, model, img_path='', save_path='data\predicted'):
    im = imageio.imread(img_path)
    path = img_path.split('.')[0]
    img_name = img_path.split('.')[-1] + "_predicted.jpg"
    if not os.path.isdir('new_folder'):
        mode = 0o777
        os.mkdir(path, mode)
    predict_path = os.path.join(save_path, img_name)
    pred = model_1(im)
    pred_df = pred.pandas().xyxy[0]
    pred_df = pred_df.loc[pred_df.confidence > 0.4]
    results = []
    i=10
    for idx, row in pred_df.iterrows():
        # a = (int(row['xmin']), int(row['ymin']))
        # b = (int(row['xmax']), int(row['ymax']))
        # c = (int(row['xmin']), int(row['ymin']) - 10)
        # print(a,b)
        # 150:225, 410:485
        ymin = math.ceil(row['ymin'])
        ymax = math.ceil(row['ymax'])
        xmin = math.ceil(row['xmin'])
        xmax = math.ceil(row['xmax'])
        
        a = (xmin, ymin)
        b = (xmax, int(row['ymax']))
        c = (int(row['xmin']), ymax - i)
        print(f'{ymin}:{ymax},{xmin}:{xmax}')
        if (ymax) >150 and (xmax)>150:
            crop_img = im[ymin:ymax, xmin:xmax]
        else:
            crop_img = im
        #try to sharpen img
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        crop_img = cv2.filter2D(crop_img, -1, kernel)
        #crop_img = im[151:223, 519:602]
        result = predict_sign_class(model, crop_img)
        results.append(result)
        # print(result)
        cv2.putText(im, result, c, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        im = cv2.rectangle(im, a, b, (0, 255, 0), 1)
        # plt.figure(figsize=(100,100))
        i+=15
    imageio.imwrite(predict_path, im)
    final_result = ','.join(results)
    return predict_path, final_result
@st.cache(suppress_st_warning=True)
def load_models():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    f_checkpoint = "best(4).pt"
    f_checkpoint_1 = "gtsrb2gtsrb_2021-11-27-144458_best.pth"
    listfile = os.listdir(dir_path)
    
    if f_checkpoint not in listfile:
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            url = 'https://drive.google.com/uc?id=1kznuazx4fO8starp_JONzAeAtWjRnZ3A'
            output = 'best(4).pt'
            gdown.download(url, output, quiet=True)
    if f_checkpoint_1 not in listfile:
        with st.spinner("Downloading model2... this may take awhile! \n Don't stop it!"):
            #download_file_from_google_drive('1i2E951VghoPQn-N742n8zE0_i3jPO-yy', f_checkpoint_1)
            url = 'https://drive.google.com/uc?id=1i2E951VghoPQn-N742n8zE0_i3jPO-yy'
            output = "gtsrb2gtsrb_2021-11-27-144458_best.pth"
            gdown.download(url, output, quiet=True)
    model_1 = torch.hub.load('ultralytics/yolov5', 'custom', path=f_checkpoint, force_reload=True) # default
    model = load_sillnet(f_checkpoint_1)
    return model_1,model

st.set_page_config(
    page_title="Traffic Sign Detection🚦",
)


@contextmanager
def st_redirect(src, dst):
    '''
        Redirects the print of a function to the streamlit UI.
    '''
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    '''
        Sub-implementation to redirect for code redability.
    '''
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    '''
        Sub-implementation to redirect for code redability in case of errors.
    '''
    with st_redirect(sys.stderr, dst):
        yield


def _all_subdirs_of(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def _get_latest_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(_all_subdirs_of(os.path.join('runs', 'detect')), key=os.path.getmtime)


parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='data\images', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
opt = parser.parse_args()
opt.weights = 'best(4).pt'
CHOICES = {0: "Image Upload"}


def _save_uploadedfile(uploadedfile):
    '''
        Saves uploaded videos to disk.
    '''
    with open(os.path.join("data", "videos", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())


def _format_func(option):
    '''
        Format function for select Key/Value implementation.
    '''
    return CHOICES[option]


inferenceSource = str(st.sidebar.selectbox('Select Source to detect:',
                      options=list(CHOICES.keys()), format_func=_format_func))

if inferenceSource == '0':
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='In progress'):
            st.sidebar.image(uploaded_file)
            picture = Image.open(uploaded_file)
            picture = picture.save(f'data/images/{uploaded_file.name}')
            opt.source = f'data/images/{uploaded_file.name}'
    else:
        is_valid = False
# else:
#     uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])
#     if uploaded_file is not None:
#         is_valid = True
#         with st.spinner(text='In progress'):
#             st.sidebar.video(uploaded_file)
#             _save_uploadedfile(uploaded_file)
#             opt.source = f'data/videos/{uploaded_file.name}'
#     else:
#         is_valid = False

st.title('Welcome to Traffic Sign Detection🚦')

inferenceButton = st.empty()

if is_valid:
    if inferenceButton.button('Launch the Detection!'):
        # model_1 = torch.hub.load('ultralytics/yolov5', 'custom', path='best(4).pt', force_reload=True) # default
        # model = load_sillnet('sillnet/gtsrb2gtsrb_2021-11-27-144458_best.pth')
        with st_stdout("info"):
            model_1,model = load_models()
            predicted_img, results = make_prediction(model_1, model, opt.source)
            i=0
            speech = f"Please Attend a head Traffic Sign {results}"
            #print(speech)
            tts = gTTS(speech)
            filename = opt.source.split('.')[0]+'.mp3'
            filename =  'data/audio/'+filename.split('/')[-1]
            
            print(filename)
            tts.save(filename)
        if inferenceSource != '0':
            st.warning('Video playback not available on deployed version due to licensing restrictions. ')
            with st.spinner(text='Preparing Video'):
                for vid in os.listdir(_get_latest_folder()):
                    st.video(f'{_get_latest_folder()}/{vid}')
                st.balloons()
        else:
            with st.spinner(text='Preparing Images'):
                st.text(results)
                st.audio(filename)
                st.image(predicted_img)
                st.balloons()
