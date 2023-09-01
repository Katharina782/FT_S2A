import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snsb

from Basenji2_torch.data_utils import *

from Basenji2_torch.basenji_architecture import *



#from architecture_batchNorm_momentum import *
model_dir = "/data/mikulik/mnt/gcs_basenj/basenji_output/clean_slate/"
pretrained_model = "basenji_paper_param_real_data_no_it_corr_0.15_sgd_4_augmentTrue_model_validation_checkpoint"
num_dilated_conv = 11
num_conv = 6
conv_target_channels = 768
dilation_rate_init = 1
bn_momentum = .9
dilation_rate_mult = 1.5
experiments_human = 5313
experiments_mouse = 1643


class BasenjiContr(nn.Module):
    def __init__(self):
        super(BasenjiContr, self).__init__()
        num_dilated_conv = 11
        num_conv = 6
        conv_target_channels = 768
        dilation_rate_init = 1
        bn_momentum = .9
        dilation_rate_mult = 1.5
        experiments_human = 5313
        experiments_mouse = 1643
        self.base_model = BasenjiModel( 
                    n_conv_layers=num_conv,
                    n_dilated_conv_layers=num_dilated_conv, 
                    conv_target_channels=conv_target_channels,
                    bn_momentum=bn_momentum,
                    dilation_rate_init=dilation_rate_init, 
                    dilation_rate_mult=dilation_rate_mult, 
                    human_tracks=experiments_human, 
                    mouse_tracks=experiments_mouse)

    def forward(self, 
                x,
                head = None,
                target_mask = None):
        
        self.target_mask = target_mask
        self.target_mask_mass = self.target_mask.sum()
        out = self.base_model(x, head)
        
        
        masked_out = (self.target_mask * out).sum() / self.target_mask_mass # compute the average signal for all experiments and bins
        return masked_out, out
        

class ComputeBasenjiContribution():
    def __init__(self, model_path, model_name, data_parallel=False):
        self.model = BasenjiContr()
        self.model.base_model.load_state_dict(torch.load(f"{model_path}{model_name}", map_location=torch.device("cpu")))
    
    def input_x_gradient(self, input_sequence, target_mask, head, plot=False):
        self.input_sequence = input_sequence
        self.target_mask = target_mask
        self.head = head
    
        self.input_sequence.requires_grad = True
        output, prediction = self.model(self.input_sequence, target_mask = self.target_mask, head = self.head)
        output.backward()
        self.input_sequence.requires_grad = False
        inp_grad = (self.input_sequence.squeeze().detach().cpu() * self.input_sequence.grad.squeeze().detach().cpu()).sum(axis=-1)
        if plot: 
            sns.lineplot(inp_grad)
            plt.show()
            return inp_grad, prediction
        return inp_grad, prediction