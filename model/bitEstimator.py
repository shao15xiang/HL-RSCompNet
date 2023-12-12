# from .basics import *
# import pickle
# import os
# import codecs
import torch.nn as nn
import torch
import torch.nn.functional as F

class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None


    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)
        
        

class BitEstimator(nn.Module):
    '''
    Estimate bit
    '''
    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        
    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)
    
    
    
    
    
    


class Bitparm_new(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(Bitparm_new, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)

class BitEstimator_new(nn.Module):
    '''
    Estimate bit
    '''
    def __init__(self, channel):
        super(BitEstimator_new, self).__init__()
        self.f1 = Bitparm_new(channel)
        self.f2 = Bitparm_new(channel)
        self.f3 = Bitparm_new(channel)
        self.f4 = Bitparm_new(channel, True)
        
    def forward(self, x):
#        print(x.shape)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
