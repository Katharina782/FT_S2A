import math

import torch.optim as optim
import itertools as it
from einops import rearrange

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from data_utils import *


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# losses and metrics


#def poisson_loss(pred, target):
#    return (pred - target * log(pred)).mean()

def poisson_loss(pred, target, reduce="mean"):
    if reduce == "mean":
        return (pred - target * log(pred)).mean()
    elif reduce == "sum":
        return (pred - target * log(pred)).sum()

# rewrite the code below as a list comprehension 

def get_filter_nums(start=288, end=768, factor=1.1776, num_conv=7):
    filter_list = []
    for i in range(1, num_conv+1):
        filter_list.append(start)
        start = int(np.round(start*factor))
    if filter_list[-1] != end:
        filter_list[-1] = 768
    return filter_list




class ConvStem(nn.Module):
    def __init__(self, filters_init=288, kernel_size=15, pool_size=2, bn_momentum=0.1):
        super(ConvStem, self).__init__()
        self.conv_stem = nn.ModuleList()
        #print(filters_init, kernel_size, pool_size)
        self.conv_stem.append(nn.Sequential(
            nn.Conv1d(in_channels=4,
                      out_channels=filters_init,
                      kernel_size=kernel_size,
                      padding="same"
                    ),
            nn.BatchNorm1d(filters_init, momentum=bn_momentum),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=pool_size)
            ))
        
    def forward(self, x):
        x = rearrange(x, 'b c l -> b l c')
        out = self.conv_stem[0](x)
        return out


class ConvLayers(nn.Module):
    def __init__(self, n_conv_layers: int, filters_init=288, filters_target=768, filter_size=5, max_pool_width=2, bn_momentum=0.1):
        super(ConvLayers, self).__init__()
        self.filter_nums = get_filter_nums(start=filters_init, end=filters_target, factor=1.1776, num_conv=n_conv_layers+1)
        self.conv_layers = nn.ModuleList()
        self.layer_dimensions = {}
        for layer in range(len(self.filter_nums) - 1):
            in_channels=self.filter_nums[layer]
            out_channels=self.filter_nums[layer + 1]
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels=in_channels, 
                          out_channels=out_channels,
                          kernel_size=filter_size,
                          padding="same"),
                nn.BatchNorm1d(out_channels, momentum=bn_momentum),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=max_pool_width))
                          )        

    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            #print(f"conv input: {x.shape}")
            out = layer(x)
            #print(f"conv output: {out.shape}")
            self.layer_dimensions[f"layer_{i}"] = out.shape#.detach().numpy().squeeze()
            #print(out.shape)
            x = out
        return x


class DilatedLayers(nn.Module):
    def __init__(self, num_dilated_conv:int, input_size:int, channel_init:int, kernel_size=3, dilation_rate_init=1, rate_mult=1.5, dropout_rate=0.3, bn_momentum=0.1):
        super(DilatedLayers, self).__init__()
        self.dilated_layers = nn.ModuleList()
        self.layer_dimensions = {}
        self.dilation_rate = dilation_rate_init
        self.rate_mult = rate_mult
        self.kernel_size = kernel_size
        self.gelu = nn.GELU()
        for _ in range(num_dilated_conv):
            self.dilation_rate *= self.rate_mult
            self.dilated_layers.append(
                nn.Sequential(
                nn.Conv1d(in_channels=input_size,
                          out_channels=channel_init,
                          kernel_size=self.kernel_size,
                          dilation=int(np.round(self.dilation_rate)),
                          padding="same"),
                nn.BatchNorm1d(channel_init, momentum=bn_momentum),
                nn.GELU(),
                nn.Conv1d(in_channels=channel_init,
                        out_channels=input_size,
                        kernel_size=1, 
                        padding="same"),
                nn.BatchNorm1d(input_size, momentum=bn_momentum),
                nn.Dropout(p=dropout_rate)
            ))

    def forward(self, x):
        for i, layer in enumerate(self.dilated_layers):
            out = layer(x)
            self.layer_dimensions[f"layer_{i}"] = [out.shape[1], out.shape[2], layer[0].dilation[0], self.kernel_size]
            x = x + out
            x = self.gelu(x)
        return x
    
        
class FinalLayers(nn.Module):
    def __init__(self, channel_init=768, out_channels=1536, target_length=896, dropout_rate=0.05, crop=64, bn_momentum=0.1):
        super(FinalLayers, self).__init__()
        self.target_length = target_length    
        self.conv = nn.ModuleList()
        self.crop = crop
        self.conv.append(nn.Sequential(
            nn.Conv1d(in_channels=channel_init,
                      out_channels=out_channels,
                      kernel_size=1,
                      padding="same"
                      ),
            nn.BatchNorm1d(out_channels, momentum=bn_momentum),
            nn.Dropout(p=dropout_rate),
            nn.GELU()
        ))
    def forward(self, x):
        out_cropped = x[:, :, self.crop:-self.crop]
        #(self.conv[0])
        out = self.conv[0].forward(out_cropped)
        assert out.shape[-1] == self.target_length
        return out
            
class OutputHeads(nn.Module):
    def __init__(self,output_heads:dict):
        super(OutputHeads, self).__init__()
        self.heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(1536, features),
            nn.Softplus()
        ), output_heads))
    def forward(self, x, head):
        x = rearrange(x, 'b c l -> b l c')
        return self.heads[head](x)



class BasenjiModel(nn.Module):
    def __init__(self, 
                 n_conv_layers:int,
                 n_dilated_conv_layers:int, 
                 conv_target_channels:int,
                 bn_momentum=0.1,
                 dilation_rate_init=1, 
                 dilation_rate_mult=2, 
                 human_tracks=5313, 
                 mouse_tracks=1643):
        super(BasenjiModel, self).__init__()
        self.n_conv_layers = n_conv_layers
        self.conv_target_channels = conv_target_channels
        self.n_dilated_conv_layers = n_dilated_conv_layers
        self.bn_momentum = bn_momentum
        self.dilation_rate_init = dilation_rate_init
        self.dilation_rate_mult = dilation_rate_mult
        self.conv_stem = ConvStem(bn_momentum=self.bn_momentum)
        self.conv_layers = ConvLayers(n_conv_layers=self.n_conv_layers, bn_momentum=self.bn_momentum)
        self.dilated_layers = DilatedLayers(num_dilated_conv=self.n_dilated_conv_layers,
                                            input_size=self.conv_target_channels,
                                            channel_init=self.conv_target_channels//2,
                                            dilation_rate_init=self.dilation_rate_init,
                                            rate_mult=self.dilation_rate_mult,
                                            bn_momentum=self.bn_momentum)
        self.final_layers = FinalLayers(bn_momentum=self.bn_momentum)
        self.output_heads = OutputHeads(output_heads=dict(human = human_tracks, mouse= mouse_tracks))

    def forward(self, sequence, head):
        stem = self.conv_stem.forward(sequence)
        embedding = self.conv_layers.forward(stem)
        out = self.dilated_layers.forward(embedding)
        final = self.final_layers.forward(out)
        #final = rearrange(final, 'b c l -> b l c')
        return self.output_heads.forward(final, head)


                            

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}