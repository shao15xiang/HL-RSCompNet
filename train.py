# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:57:50 2021

@author: 28004
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from Meter import AverageMeter
import os
import torchvision
from MSSSIM import MSSSIM
from ourloader import RGBdataset

from utils.torch_utils import  intersect_dicts
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler
from model.CompNet import CompNet


"""

The authors are very grateful to the GitHub contributors who provided some of the source 
code for this research.

"""

warmup_step = 0
cur_lr = 0.0001
print_freq = 50
global_step = 0
decay_interval = 1000
lr_decay = 0.1



class Trainer:
    def __init__(self, trainloader, testloader, model,args):
        self.model = model.cuda()
        self.loss_func = nn.MSELoss()
                
        self.trainloader = trainloader
        self.testloader = testloader
        self.optim = torch.optim.Adam(self.model.parameters(), lr = args.lr)
        self.psnr, self.mse, self.msssim, self.bpp= [AverageMeter(print_freq) for _ in range(4)]
        self.psnr_val, self.mse_val, self.msssim_val,self.bpp_val = [AverageMeter(print_freq) for _ in range(4)]
        self.best_psnr = 0.
        self.max_bpp = 100
        self.args = args
        self.MSSSIM = MSSSIM()
        self.global_step =  global_step
        self.scheduler = LR_Scheduler('step', num_epochs = args.epoch, base_lr = args.lr, lr_step=args.step)
        
    def adjust_learning_rate(self, optimizer):
        global cur_lr
        global warmup_step
        if self.global_step < warmup_step:
            lr = self.args.lr * global_step / warmup_step
        elif self.global_step < decay_interval:
            lr = self.args.lr
        else:
             lr = self.args.lr * (lr_decay ** (global_step // decay_interval))
        cur_lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
    def train(self, epoch, total_epoch):
        self.model.train()
        with tqdm(total=len(self.trainloader)) as tbar: 
            for i,data in enumerate(self.trainloader):    
                
                self.global_step += 1
                
                img = data.cuda()
                recon_img, mse_loss, bpp_loss, bpp_z, bpp = self.model(img)
                self.scheduler(self.optim, i, epoch, self.best_psnr)                
                
                psnr = -10*torch.log10(self.loss_func(img, recon_img))
                                
                sim = self.MSSSIM(img, recon_img)                
                
                loss = mse_loss + bpp_loss 

                self.optim.zero_grad()
                loss.backward()
                
                def clip_gradient(optimizer, grad_clip):
                    for group in optimizer.param_groups:
                        for param in group["params"]:
                            if param.grad is not None:
                                param.grad.data.clamp_(-grad_clip, grad_clip)
                clip_gradient(self.optim, 5)
                        
                self.optim.step()
                self.mse.update(loss.item())
                self.psnr.update(psnr.item())
                self.msssim.update(sim.item())
                self.bpp.update(bpp.item())
                tbar.set_description('Train loss: %.3f  PSNR: %.3f MSSSIM: %.3f bpp: %.3f' % (self.mse.avg, self.psnr.avg, 
                                                                                              self.msssim.avg, self.bpp.avg, ))
                tbar.update()
                                   
   
    def test(self, epoch):
        self.model.eval()
        
        with tqdm(total=len(self.testloader)) as tbar: 
            for i,data in enumerate(self.testloader):
                img = data.cuda()
                with torch.no_grad():
                    recon_img, mse_loss, bpp_loss, bpp_z, bpp = self.model(img)
                    
                loss = mse_loss +  bpp_loss    
                
                psnr = -10*torch.log10(self.loss_func(img, recon_img))
                
                sim = self.MSSSIM(img, recon_img)
                self.mse_val.update(loss.item())
                self.psnr_val.update(psnr.item())  
                self.msssim_val.update(sim.item())  
                self.bpp_val.update(bpp.item())  
                
                tbar.set_description('Test loss: %.3f  PSNR: %.3f MSSSIM: %.3f bpp: %.3f' % (self.mse_val.avg, self.psnr_val.avg, self.msssim_val.avg, self.bpp_val.avg))
                
                tbar.update()
                
        self.save_model(epoch, self.args.save_images)
        
        if  self.psnr_val.avg > self.best_psnr:
            
            if self.args.vis:
                for b in range(recon_img.shape[0]):
                    reimg = recon_img[b][:3,:,:].cpu() 
                        
                    reimg = torchvision.transforms.functional.to_pil_image(reimg)
                    origin = img[b][:3,:,:].cpu() 
                    origin = torchvision.transforms.functional.to_pil_image(origin)
                                    
                    reimg.save('{}/{}_{}_recon.png'.format(self.args.save_images, str(b),str(b)))
                    origin.save('{}/{}_{}_ori.png'.format(self.args.save_images, str(b),str(b)))                
                
                        
            self.max_bpp = self.bpp_val.avg
            self.best_psnr = self.psnr_val.avg
            
            if not self.args.test:
                self.save_model("best", self.args.save_images)
            print('-** weight saved in ',self.args.save_images)
            
            
    def save_model(self, iter, name):
        if not os.path.exists(name): os.mkdir(name)
        torch.save(self.model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--pretrained", default = "")
    parser.add_argument("--save_images", default = "weights/")
        
    parser.add_argument('--lr', default=0.0001, type=int, help='initial learning rate for model training')
    parser.add_argument('--epoch', default=600, type=int, help='model training epoch')
    parser.add_argument('--startepoch', default=0, type=int, help='model training epoch')
    parser.add_argument('--step', default=200, type=int, help='model training step')
    parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
    parser.add_argument('--out_channel_N', default=192, type=int, help='out_channel_N')
    parser.add_argument('--out_channel_M', default=320, type=int, help='out_channel_M')
    
    parser.add_argument("--trainpath", default = "your path/train/images",  help='training dataset path')
    parser.add_argument("--testpath", default = "your path/val/images",  help='validation dataset path')    
            
    parser.add_argument('--vis',  default=True, type=bool, help='print image')
    parser.add_argument('--test', default=False, type=bool, help='model test')
        
    args = parser.parse_args()
    
    train_set = RGBdataset(root = args.trainpath, mode = "train")
    
    trainloader = DataLoader(train_set, batch_size=args.batchsize, num_workers=8, shuffle=True)
    
    test_set = RGBdataset(root  =  args.testpath,  mode = "val")
    
    testloader = DataLoader(test_set, batch_size=args.batchsize, num_workers=8, shuffle=False)
    torch.backends.cudnn.enabled = True
    
    
    if args.save_images != '':
        os.makedirs(args.save_images, exist_ok=True)
    
    out_channel_N, out_channel_M = args.out_channel_N,  args.out_channel_M

    model = CompNet(out_channel_N,  out_channel_M, lamb = 8192)    
    
    if args.pretrained != '':
        state_dict = torch.load(args.pretrained)
        state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect
        model.load_state_dict(state_dict, strict=args.test)  # load

    print('Start ...')
    train_val = Trainer(trainloader, testloader, model, args)
    
    if args.test:
        train_val.test(1)
    else:
        for epoch in range(args.startepoch, args.epoch):
            train_val.train(epoch, args.epoch)
            train_val.test(epoch)
