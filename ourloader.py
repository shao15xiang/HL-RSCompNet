#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:15:03 2022

@author: xs
"""

import os,cv2,torch

import numpy as np
from torch.utils import data
import random 

def RandomCrop(img, size = (256,256)):
    w, h, _ = img.shape
   
    th, tw = size
    x1 = random.randint(0, w - tw - 1)
    y1 = random.randint(0, h - th - 1)
    
    return img[x1:x1 + tw, y1:y1 + th, :]
       
def CenterCrop(img, size = (256,256)):
    w, h, _ = img.shape
    th, tw = size    
    x1 = (w - tw - 1) // 2
    y1 = (h - th - 1) //2
    
    return img[x1:x1 + tw, y1:y1 + th, :]
    
        
class RGBdataset(data.Dataset):
    def __init__(self, root, mode = "train", img_name = None):
        self.root = root
        self.mode = mode
        self.images = os.listdir(root)
        self.singel_name = img_name
        
        if len(self.images) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        
    def __getitem__(self, index):
            
        img_name = self.images[index]        
        
        img = cv2.imread(self.root + os.sep + img_name) 
        
        if self.singel_name:
            img = cv2.imread(self.root + os.sep + self.singel_name) 
            
        img_np = np.asarray(img)
        
        if self.mode == "train" : 
            img_np = RandomCrop(img_np)
            
        elif self.mode == 'val':  
            img_np = CenterCrop(img_np)
        
        else: 
            img_np = img_np
                    
        img_np = np.transpose(img_np, (2,0,1))
        
        img_tensor = torch.from_numpy(img_np)
        
        return img_tensor / 255.0

    def __len__(self):
        return len(self.images)
        
    
