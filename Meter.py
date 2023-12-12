# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:38:51 2021

@author: 28004
"""
from typing import List

class AverageMeter:
    def __init__(self, length: int, name: str = None):
        assert length > 0
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.current: int = -1
        self.history: List[float] = [None] * length

    @property
    def val(self) -> float:
        return self.history[self.current]

    @property
    def avg(self) -> float:
        return self.sum / self.count
    
    def reset(self) -> float:
        self.sum = 0
        self.count =0

    def update(self, val: float):
        self.current = (self.current + 1) % len(self.history)
        self.sum += val

        old = self.history[self.current]
        if old is None:
            self.count += 1
        else:
            self.sum -= old
        self.history[self.current] = val


#import torch
#import torch.nn as nn
#from MSSSIM import MSSSIM
#
#class AverageMeter(object):
#    def __init__(self,):
#        self.count = 0
#        self.mse = nn.MSELoss()
#        self.MSSSIM = MSSSIM()
#        self.psnr = 0
#        self.sim = 0
#
#    def add_batch(self, img, reco_img):
#        bs = img.shape[0]
#        assert img.shape == reco_img.shape
#        self.psnr = -10*torch.log10(self.loss_func(img, reco_img)) / bs + self.psnr
#        self.sim = self.MSSSIM(img, reco_img) / bs + self.sim
#        self.count += 1
# 
#    def reset(self):
#        self.count =0
#        self.psnr = 0
#        self.sim = 0
#    def avg(self):
#        return self.psnr / self.count, self.sim / self.count
#
#
