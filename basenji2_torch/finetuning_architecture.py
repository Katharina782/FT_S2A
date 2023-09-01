import torch
from architecture_batchNorm_momentum import *
from data_utils_finetuning import *
from data_augmentation import *
import torch.optim as optim



class Pretrained(nn.Module):
    def __init__(self, num_conv, num_dilated_conv, conv_target_channels, bn_momentum, dilation_rate_init, dilation_rate_mult, experiments_human, experiments_mouse):
        super(Pretrained, self).__init__()
        self.pretrained_model = BasenjiModel( 
                    n_conv_layers=num_conv,
                    n_dilated_conv_layers=num_dilated_conv, 
                    conv_target_channels=conv_target_channels,
                    bn_momentum=bn_momentum,
                    dilation_rate_init=dilation_rate_init, 
                    dilation_rate_mult=dilation_rate_mult, 
                    human_tracks=experiments_human, 
                    mouse_tracks=experiments_mouse)
        
    def forward(self, x):
        stem = self.pretrained_model.conv_stem.forward(x)
        conv = self.pretrained_model.conv_layers.forward(stem)
        dil = self.pretrained_model.dilated_layers.forward(conv)
        #print(stem.shape, conv.shape, dil.shape)
        embedding = self.pretrained_model.final_layers.forward(dil)
        return embedding

            
class OutputHeadsNew(nn.Module):
    def __init__(self, tracks:int):
        super(OutputHeadsNew, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(1536, tracks),
            nn.Softplus()
        )
    def forward(self, x):
        x = rearrange(x, 'b c l -> b l c')
        return self.head(x)

class FineTuning(nn.Module):
    def __init__(self, data_dir, 
                model_name,
                tracks,
                num_conv,
                num_dilated_conv,
                conv_target_channels, 
                bn_momentum, dilation_rate_init, 
                dilation_rate_mult, 
                experiments_human, 
                experiments_mouse):
        super(FineTuning, self).__init__()
        self.model = Pretrained(num_conv,
                num_dilated_conv,
                conv_target_channels, 
                bn_momentum, dilation_rate_init, 
                dilation_rate_mult, 
                experiments_human, 
                experiments_mouse)
        self.model.pretrained_model.load_state_dict(torch.load(os.path.join(data_dir, f"{model_name}.pt"), map_location=torch.device("cpu")))
        self.new_head = OutputHeadsNew(tracks)

    def forward(self, sequence):
        embedding = self.model.forward(sequence)
        return self.new_head(embedding)
                                    

