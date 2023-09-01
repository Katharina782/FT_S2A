import os
import pickle
import pandas as pd
import numpy as np
import tqdm as tqdm
from optparse import OptionParser

#import math
import torch
import torch.nn.functional as F
#from torch.utils.checkpoint import checkpoint_sequential
import torch.optim as optim

#from data import str_to_one_hot, seq_indices_to_one_hot

from architecture_batchNorm_momentum import *
from finetuning_architecture import * 
#from original_data_preprocessing import *
from metrics import *
from evaluate import * 
from gastrulation_correlation_tss import *
from data_utils_finetuning import *



usage = "usage: %prog [options] <file_name> <data_dir> <checkpoint_dir>"
parser = OptionParser(usage)
parser.add_option("-l", dest="seq_length",
                  default=131072, type="int",
                  help="Input sequence length [Default: %default]")
parser.add_option("--species", dest="species",
                  default="human",
                  help="Species for which to compute correlations. One of human or mouse.")
#parser.add_option("--subset", dest="subset",
#                  default="valid",
#                  help="One of train, valid or test.")
parser.add_option("--state_dict", dest="state_dict",
                  default=True,
                  help="True if the model was saved as a state dictionary and DataParallel.")
parser.add_option("--momentum", dest="momentum",
                  default=0.99, type="float",
                  help="Momentum for optimization [Default: %default]")          
parser.add_option("--dil_mult", dest="dilation_rate_mult",
                  default=1.5, type="float",
                  help="Factor of dilation rate increase [Default: %default]")     
parser.add_option("--bn_momentum", dest="bn_momentum",
                  default=0.9, type="float",
                    help="Batch norm momentum [Default: %default]")
parser.add_option("--rna", dest="rna_tracks",
                  default=37, type="int",
                  help="Number of output tracks for rna [Default: %default]")
parser.add_option("--atac", dest="atac_tracks",
                  default=35, type="int",
                  help="Number of output tracks for atac [Default: %default]")

(options, args) = parser.parse_args()

if len(args) != 3:
    parser.error("Must provide file name, data directory, checkpoint directory and overlap_tss file")
else:
    pretrained_model = args[0]
    data_dir = args[1]
    checkpoint_dir = args[2]


print(f"seq-len: {options.seq_length}, species: {options.species}, file_name: {pretrained_model}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")



# initialize model for finetuning
tracks =options.rna_tracks + options.atac_tracks
# tracks = 72 # both rna and atac tracks
#pretrained_model =# "basenji_paper_param_real_data_no_it_corr_0.15_sgd_4_augmentTrue_model_validation_checkpoint"
num_dilated_conv = 11
num_conv = 6
conv_target_channels = 768
dilation_rate_init = 1
bn_momentum = .9
dilation_rate_mult = 1.5
experiments_human = 5313
experiments_mouse = 1643



model = FineTuning(data_dir=os.path.join(data_dir),
                            model_name=pretrained_model, 
                            tracks = tracks,                
                            num_conv=num_conv,
                            num_dilated_conv=num_dilated_conv,
                            conv_target_channels=conv_target_channels, 
                            bn_momentum=0.9, 
                            dilation_rate_init=dilation_rate_init, 
                            dilation_rate_mult=1.5, 
                            experiments_human=5313, 
                            experiments_mouse=1643).to(device)





overlap = pd.read_csv(os.path.join(data_dir,'mouse_gastrulation_tss_region_overlap.csv'), header=0)
overlap.drop_duplicates(subset="gene_id", keep="first", inplace=True)
# Correlation at TSS 
count_mat = np.load(os.path.join(data_dir, "rna_lib_scaled.npy"))            



for file in os.listdir(os.path.join(data_dir, "finetune_output/rna/")):
    if file.endswith("_model_validation_checkpoint.pt") & file.startswith("finetune_0.00"):
        name = file.split("_model_validation_checkpoint.pt")[0]
        print(name)
        model.load_state_dict(torch.load(os.path.join(data_dir, "finetune_output/rna/", f"{file}"), map_location=torch.device("cpu")))#, strict=False)
        for subset in ["train", "test", "valid"]:
            gene_dict = collect_counts_tss(data_dir, seq_length=131_072, 
                                            subset=subset, model=model, 
                                            device=device,overlap=overlap, 
                                            count_mat=count_mat,joint_training=False)
            with open(os.path.join(checkpoint_dir, f"{name}_{subset}_gene_dict.pkl"), "wb") as f:
                pickle.dump(gene_dict, f)



