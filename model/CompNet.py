#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 08:47:00 2023

@author: xs
"""

import sys
sys.path.append('..')

import math
import torch.nn as nn

import torch
from .DWT import DWT, IWT
from .GDN import GDN
from .bitEstimator import BitEstimator

"""

The authors are very grateful to the GitHub contributors who provided some of the source 
code for this research.


Reference: https://github.com/LiuLei95/PyTorch-Learned-Image-Compression-with-GMM-and-Attention

"""

class DualGMMEntropy(nn.Module):
    def __init__(self, num_filters=128, kernel = (5,5)):
        super(DualGMMEntropy, self).__init__()
        self.maskedconv = CheckboardMaskedConv2d(
            num_filters * 2, num_filters*2, 5, stride=1, padding=2)
        
        torch.nn.init.xavier_uniform_(self.maskedconv.weight.data, gain=1)
        torch.nn.init.constant_(self.maskedconv.bias.data, 0.0)
        
        self.conv1 = nn.Conv2d(num_filters*4, 640, 1, stride=1)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(640, 640, 1, stride=1)
        self.leaky_relu2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv2d(640, 640 * 2, 1, stride=1)
        self.leaky_relu3 = nn.LeakyReLU()
        self.convh = nn.Conv2d(640, num_filters * 9, 1, stride=1, bias=False)
        self.convl = nn.Conv2d(640, num_filters * 9, 1, stride=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.dwt = DWT()

    def forward(self, x_l, x_h, y_l, y_h):
        
        y = torch.cat((y_l, y_h), 1)
        y = self.maskedconv(y)
        sigma = torch.cat((x_l, x_h), 1)
        # print(y.shape)
        x = torch.cat([y, sigma], dim=1)
        # print(x.shape)
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.leaky_relu3(self.conv3(x))
        
        x_l, x_h = torch.split(x, split_size_or_sections=int(x.shape[1]/2), dim=1)
        
        x_l = self.convl(x_l)
        x_h = self.convh(x_h) 
        
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = \
            torch.split(x_l, split_size_or_sections=int(x_l.shape[1]/9), dim=1)            
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        scale2 = torch.abs(scale2)
        probs = torch.stack([prob0, prob1, prob2], dim=-1) 
        
        probs_l = self.softmax(probs)
        means_l = torch.stack([mean0, mean1, mean2], dim=-1)
        variances_l = torch.stack([scale0, scale1, scale2], dim=-1)
        
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = \
            torch.split(x_h, split_size_or_sections=int(x_h.shape[1]/9), dim=1)
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        scale2 = torch.abs(scale2)
        probs = torch.stack([prob0, prob1, prob2], dim=-1)    
        
        probs_h = self.softmax(probs)
        means_h = torch.stack([mean0, mean1, mean2], dim=-1)
        variances_h = torch.stack([scale0, scale1, scale2], dim=-1)
            
        return [means_l, variances_l, probs_l], [means_h, variances_h, probs_h]


class CheckboardMaskedConv2d(nn.Conv2d):
    """
    Reference: https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)
        return out


class Enc(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M = 192):
        super(Enc, self).__init__()
        
        self.dwt = DWT()
        
        self.conv1  = nn.Conv2d(out_channel_N, out_channel_N // 4, 3, 1, 1)
        self.conv2  = nn.Conv2d(out_channel_N, out_channel_N // 4, 3, 1, 1)
        
        self.encoder_l =  nn.Conv2d(out_channel_N // 2, out_channel_M, 3, stride=1, padding=1)        
        self.encoder_h =  nn.Conv2d(6 * out_channel_N // 4, out_channel_M, 3, stride=1, padding=1)

    def forward(self, l, h):
        
        l = self.conv1(l)
        h = self.conv2(h)
        
        l_ll, l_hl, l_lh, l_hh = self.dwt(l)
        h_ll, h_hl, h_lh, h_hh = self.dwt(h)
        
        x_l = torch.cat((l_ll, h_ll), 1)
        
        x_h = torch.cat((l_hl, h_hl, l_lh , h_lh, l_hh , h_hh), 1)
        
        x_l = self.encoder_l(x_l)
        x_h = self.encoder_h(x_h)
        
        return x_l, x_h
    



class Dec(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M = 192):
        super(Dec, self).__init__()
        self.dwt = DWT()
        self.upl = nn.ConvTranspose2d(out_channel_N, out_channel_M // 4, 3, stride=2, padding=1, output_padding = 1)
        self.uph = nn.ConvTranspose2d(out_channel_N, out_channel_M // 4, 3, stride=2, padding=1, output_padding = 1)
        
        self.dec_l = nn.ConvTranspose2d(out_channel_M // 2, out_channel_M, 3, stride=2, padding=1, output_padding = 1)
        self.dec_h = nn.ConvTranspose2d(6 * out_channel_M // 4, out_channel_M, 3, stride=2, padding=1, output_padding = 1)

    def forward(self, l, h):
        
        l =self.upl(l)
        h =self.uph(h)
        
        l_ll, l_hl, l_lh, l_hh = self.dwt(l)
        h_ll, h_hl, h_lh, h_hh = self.dwt(h)
        
        x_l = torch.cat((l_ll, h_ll), 1)
        x_h = torch.cat((l_hl, h_hl, l_lh , h_lh, l_hh , h_hh), 1)
        
        x_l = self.dec_l(x_l)
        x_h = self.dec_h(x_h)
        
        return x_l, x_h


class Analysis_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, inpband = 4):
        super(Analysis_net, self).__init__()
        
        self.conv = nn.Conv2d(inpband, out_channel_N, 5, 2, padding = 2)
        self.gdn = GDN(out_channel_N)
        
        self.first_conv_l = nn.Conv2d(out_channel_N, out_channel_N, 5, 2, padding = 2)
        self.first_conv_h = nn.Conv2d(out_channel_N, out_channel_N, 5, 2, padding = 2)
        self.gdn1 = GDN(out_channel_N)
        self.gdn2 = GDN(out_channel_N)
                
        self.enc1 = Enc(out_channel_N, out_channel_N)   
        self.gdn3_l = GDN(out_channel_N)
        self.gdn3_h = GDN(out_channel_N)
        self.enc2 = Enc(out_channel_N, out_channel_M)        
        
    def forward(self, x):
        
        x = self.gdn(self.conv(x))
        l = self.gdn1(self.first_conv_l(x))
        h = self.gdn2(self.first_conv_h(x))
                
        x_l, x_h = self.enc1(l, h)
        x_l = self.gdn3_l(x_l)
        x_h = self.gdn3_h(x_h)
        x_l, x_h = self.enc2(x_l, x_h)
        return x_l, x_h
    
    
       
class Synthesis_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, outband = 4):
        super(Synthesis_net, self).__init__()
        
        self.deconv1 = Dec(out_channel_M, out_channel_N)
        self.igdn1_l = GDN(out_channel_N, inverse = True)
        self.igdn1_h = GDN(out_channel_N, inverse = True)
        
        self.deconv2 = Dec(out_channel_N, out_channel_N)
        self.igdn2_l = GDN(out_channel_N, inverse = True)
        self.igdn2_h = GDN(out_channel_N, inverse = True)
        
        self.dec1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding = 1)
        self.dec2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding = 1)
        self.igdn1 = GDN(out_channel_N, inverse = True)
        self.igdn2 = GDN(out_channel_N, inverse = True)
        
        self.dec3 = nn.ConvTranspose2d(out_channel_N * 2, outband, 5, stride=2, padding=2, output_padding = 1)
        
    def forward(self, x_l, x_h):
                
        x_l, x_h = self.deconv1(x_l, x_h)
        x_l = self.igdn1_l(x_l)
        x_h = self.igdn1_h(x_h)
        
        x_l, x_h = self.deconv2(x_l, x_h) 
        x_l = self.igdn2_l(x_l)
        x_h = self.igdn2_h(x_h)
        
        x_l = self.igdn1(self.dec1(x_l))
        x_h = self.igdn2(self.dec2(x_h))
        
        x = self.dec3(torch.cat((x_l, x_h), 1))
        
        return x



class Analysis_prior_net(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Analysis_prior_net, self).__init__()
        
        self.enc1 = Enc(out_channel_M, out_channel_N)   
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.enc2 = Enc(out_channel_N, out_channel_N)
        
    def forward(self, x_l, x_h):
        
        x_l = torch.abs(x_l)
        x_h = torch.abs(x_h)
        
        x_l, x_h = self.enc1(x_l, x_h)
        
        x_l = self.relu1(x_l)
        x_h = self.relu2(x_h)
        
        x_l, x_h = self.enc2(x_l, x_h)
        return  x_l, x_h
    
    

class Synthesis_prior_net(nn.Module):
    '''
    Decode synthesis prior
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Synthesis_prior_net, self).__init__()
        
        self.deconv1 = Dec(out_channel_N, out_channel_N)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.deconv2 = Dec(out_channel_N, out_channel_M)
                        
    def forward(self, x_l, x_h):
        
        x_l, x_h = self.deconv1(x_l, x_h)
        x_l = self.relu1(x_l)
        x_h = self.relu2(x_h)
        x_l, x_h = self.deconv2(x_l, x_h)         
        
        return  x_l, x_h
    
    
    
class CompNet(nn.Module):
    def __init__(self, out_channel_N=128, out_channel_M=192, lamb=2048, band=3):
        super(CompNet, self).__init__()
        
        self.Encoder = Analysis_net(out_channel_N = out_channel_N, out_channel_M = out_channel_M, inpband = band)
        self.Decoder = Synthesis_net(out_channel_N = out_channel_N, out_channel_M = out_channel_M, outband = band)

        self.priorEncoder = Analysis_prior_net(out_channel_N = out_channel_N, out_channel_M = out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N = out_channel_N, out_channel_M = out_channel_M)        
        
        self.bitEstimator_z = BitEstimator(out_channel_N) 
        self.bitEstimator_h = BitEstimator(out_channel_N) 
        
        self.entropy = DualGMMEntropy(out_channel_M)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M
        
        self.lamb = lamb
        self.iwt = IWT()
        self.dwt = DWT()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')   
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')   

    def forward(self, input_image):
        
        batch_size = input_image.size()[0]
        
        device = torch.device("cuda:0" if input_image.is_cuda else "cpu")

        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, 
                                    input_image.size(3) // 64, 
                                    input_image.size(3) // 64).to(device)
        
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        quant_noise_x = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(3) // 16,
                                    input_image.size(3) // 16).to(device)
        quant_noise_x = torch.nn.init.uniform_(torch.zeros_like(quant_noise_x), -0.5, 0.5)
        quant_noise_z_h = torch.zeros(input_image.size(0), self.out_channel_N, 
                                    input_image.size(3) // 64, 
                                    input_image.size(3) // 64).to(device)
        
        quant_noise_z_h = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z_h), -0.5, 0.5)
        quant_noise_x_h = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(3) // 16,
                                    input_image.size(3) // 16).to(device)
        quant_noise_x_h = torch.nn.init.uniform_(torch.zeros_like(quant_noise_x_h), -0.5, 0.5)
        
        x_l, x_h = self.Encoder(input_image)
        z_l, z_h = self.priorEncoder(x_l, x_h)
        
        if self.training:
            
            compressed_x_l = x_l + quant_noise_x
            compressed_z_l = z_l + quant_noise_z
            
            compressed_z_h = z_h + quant_noise_z_h
            compressed_x_h = x_h + quant_noise_x_h
            
        else:
            compressed_x_l = torch.round(x_l)
            compressed_z_l = torch.round(z_l)
            
            compressed_z_h = torch.round(z_h)
            compressed_x_h = torch.round(x_h)
            
        z_hat_l, z_hat_h = self.priorDecoder(z_l, z_h)    

        
        
        [mean_l, sigma_l, prob_l], [mean_h, sigma_h, prob_h] = self.entropy(z_hat_l, z_hat_h,
                                                                compressed_x_l, compressed_x_h)
        
        # decode
        recon_image  = self.Decoder(compressed_x_l, compressed_x_h)
        
        mse_loss = self.lamb * torch.mean((recon_image - input_image).pow(2)) 
                
        clipped_recon_image = recon_image.clamp(0, 1)        
                
        im_shape = input_image.size()

        def feature_probs_based_GMM(feature, means, sigmas, weights):
            mean1 = torch.squeeze(means[:, :, :, :, 0])
            mean2 = torch.squeeze(means[:, :, :, :, 1])
            mean3 = torch.squeeze(means[:, :, :, :, 2])
            sigma1 = torch.squeeze(sigmas[:, :, :, :, 0])
            sigma2 = torch.squeeze(sigmas[:, :, :, :, 1])
            sigma3 = torch.squeeze(sigmas[:, :, :, :, 2])

            weight1, weight2, weight3 = torch.squeeze(weights[:, :, :, :, 0]), torch.squeeze(weights[:, :, :, :, 1]), torch.squeeze(weights[:, :, :, :, 2])
            sigma1, sigma2, sigma3 = sigma1.clamp(1e-10, 1e10), sigma2.clamp(1e-10, 1e10), sigma3.clamp(1e-10, 1e10)
            gaussian1 = torch.distributions.laplace.Laplace(mean1, sigma1)
            gaussian2 = torch.distributions.laplace.Laplace(mean2, sigma2)
            gaussian3 = torch.distributions.laplace.Laplace(mean3, sigma3)
            prob1 = gaussian1.cdf(feature + 0.5) - gaussian1.cdf(feature - 0.5)
            prob2 = gaussian2.cdf(feature + 0.5) - gaussian2.cdf(feature - 0.5)
            prob3 = gaussian3.cdf(feature + 0.5) - gaussian3.cdf(feature - 0.5)

            probs = weight1 * prob1 + weight2 * prob2 + weight3 * prob3
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs
        
        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob
        def iclr18_estimate_bits_h(z):
            prob = self.bitEstimator_h(z + 0.5) - self.bitEstimator_h(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob
        
        total_bits_l, _ = feature_probs_based_GMM(compressed_x_l, mean_l, sigma_l, prob_l)
        total_bits_h, _ = feature_probs_based_GMM(compressed_x_h, mean_h, sigma_h, prob_h)
        
        total_bits_feature = iclr18_estimate_bits_z(compressed_z_l)[0] + iclr18_estimate_bits_h(compressed_z_h)[0]
    
        bpp_feature = (total_bits_l + total_bits_h) / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])

        bpp = bpp_feature + bpp_z
        
        bpp_loss = bpp 

        return clipped_recon_image, mse_loss, bpp_loss, bpp_z, bpp

 
if __name__ == "__main__":

    inp = torch.rand(2, 3, 256, 256).cuda()

    test = CompNet(128, 192, band=3).cuda()
    recon_img, mse_loss, bpp_feature, bpp_z, bpp = test(inp)
    print(recon_img.shape,  bpp_feature.item(),  bpp.item())
