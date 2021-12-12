import sys
sys.path.append('/sillnet')
from argparse import ArgumentParser
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import math
import os
from pathlib import Path
import time
import random
import scipy.misc as m
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#from sillnet.loader import get_loader, get_data_path



def transform(img,img_size):
    img = img.astype(np.float64)
    #img -= self.mean
    if img_size is not None:
        img = cv2.resize(img, (img_size[1], img_size[0]), interpolation = cv2.INTER_AREA)
    # Resize scales images from 0 to 255, thus we need
    # to divide by 255.0
    img = img.astype(float) / 255.0
    
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.reshape(img,(1,3,64,64))
    img = torch.from_numpy(img).float().to(torch.device('cpu'))
    return img
def predict_sign_class(molel,img):
    class_gt = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
  
    if isinstance(img, str):
        img = m.imread(img)
    plt.imshow(img)
    img = np.array(img, dtype=np.uint8)
    img = transform(img,(64,64))
    feat_sem, feat_illu, feat_nowarp = net.extract(img,is_warping=True)
    feature_exc = torch.cat((feat_sem, feat_illu), 1)
    #class for dataset GTSRB
    prob, pred_sup = torch.max(net.classify(feature_exc), 1)
    #class for dataset TT100K
    prob_1, pred_sup_1 = torch.max(net.classify2(feature_exc), 1)
    if prob_1 >= prob:
        result = class_gt[pred_sup_1.tolist()[0]]
    else:
        result = class_gt[pred_sup.tolist()[0]]
    print(pred_sup,pred_sup_1)
    return result
def main():
  model = torch.load('./sillnet/sillnet_model.pth',map_location=torch.device('cpu'))
  return model

if __name__ == '__main__':
    main()