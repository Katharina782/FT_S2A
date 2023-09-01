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
parser.add_option("--joint", dest="joint",
                  default=True, type="bool",
                  help="Whether model was fine-tuned jointly on RNA and ATAC [Default: %default]")

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
num_dilated_conv = 11
num_conv = 6
conv_target_channels = 768
dilation_rate_init = 1
bn_momentum = .9
dilation_rate_mult = 1.5
experiments_human = 5313
experiments_mouse = 1643

model = FineTuning(data_dir=data_dir,
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





# dataframe contianing coordinates where gene TSSs overlap the region
overlap = pd.read_csv(os.path.join(data_dir,'mouse_gastrulation_tss_region_overlap.csv'), header=0)
overlap.drop_duplicates(subset="gene_id", keep="first", inplace=True)
# Count matrix cells x genes
count_mat = np.load(os.path.join(data_dir, "rna_lib_scaled.npy"))            


finetune_dir = os.path.join(data_dir, "finetune_output/joint/")
names = []
for file in os.listdir(finetune_dir):
    #if file.endswith("model_validation_checkpoint.pt") & file.startswith("joint") & ~("weight" in file):
    #if file.endswith("model_validation_checkpoint.pt") & file.startswith("joint_1e"):# & ("weight" in file):
    if file.endswith("model_validation_checkpoint.pt") & file.startswith("joint"):
        name = file.split("_model_validation_checkpoint.pt")[0]
        print(f"model:{name}")
        # load the fine-tuned model
        model.load_state_dict(torch.load(os.path.join(finetune_dir, f"{file}"), map_location=device))#, strict=False)
        # for each subset of the data
        for subset in ["train", "test", "valid"]:
            # get the gene dict of observed and predicted counts at the TSS of a gene
            gene_dict = collect_counts_tss(data_dir, seq_length=131_072, 
                                            subset=subset, model=model, 
                                            device=device,overlap=overlap, 
                                            count_mat=count_mat,joint_training=True)
            # save the gene dictionary for the current subset
            with open(os.path.join(checkpoint_dir, f"{name}_{subset}_gene_dict.pkl"), "wb") as f:
                pickle.dump(gene_dict, f)




#