import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snsb

from data_utils import *

from architecture_linear import *


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.base_model = nn.DataParallel(
            Enformer.from_hparams(
            dim = 1536,
            depth = 5,
            heads = 8,
            use_checkpointing=True,
            output_heads = dict(human = 5313, mouse= 1643),
            target_length = 896,
            )
        )

    def forward(self, 
                x,
                target = None,
                return_corr_coef = False,
                return_embedding = False, 
                return_only_embedding = False,
                return_loss_per_bp = False,
                head = None,
                target_mask = None):
        
        #x.requires_grad = True
        self.target_mask = target_mask
        self.target_mask_mass = self.target_mask.sum()
        out = self.base_model(x)[head]
        
        
        masked_out = (self.target_mask * out).sum() / self.target_mask_mass # compute the average signal for all experiments and bins
        return masked_out, out
        #return (softmax_profile.detach()*profile_shapes).sum(axis=-1).squeeze().mean(axis=-1)
        
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.base_model = Enformer.from_hparams(
            dim = 1536,
            depth = 5,
            heads = 8,
            use_checkpointing=True,
            output_heads = dict(human = 5313, mouse= 1643),
            target_length = 896,
            )
        

    def forward(self, 
                x,
                target = None,
                return_corr_coef = False,
                return_embedding = False, 
                return_only_embedding = False,
                return_loss_per_bp = False,
                head = None,
                target_mask = None):
        
        #x.requires_grad = True
        self.target_mask = target_mask
        self.target_mask_mass = self.target_mask.sum()
        out = self.base_model(x)[head]
        
        
        masked_out = (self.target_mask * out).sum() / self.target_mask_mass # compute the average signal for all experiments and bins
        return masked_out, out

class ComputeContribution():
    def __init__(self, model_path, model_name, data_parallel=False):#, device="cpu"):
        if data_parallel:
            self.model = CustomModel()
        else:
            self.model = LinearModel()
        #self.device = device
        #print(self.device)
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
    

