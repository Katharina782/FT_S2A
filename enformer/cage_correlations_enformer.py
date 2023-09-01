import os
import sys
import json
import functools
import pickle
import glob
import gzip
import pandas as pd
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import torch.optim as optim

from data import str_to_one_hot, seq_indices_to_one_hot

from config_enformer import EnformerConfig
#from original_data_preprocessing import *
from metrics import *
from data_utils import *




usage = "usage: %prog [options] <file_name> <data_dir> <checkpoint_dir> <overlap_file"
parser = OptionParser(usage)
parser.add_option("-l", dest="seq_length",
                  default=196608, type="int",
                  help="Input sequence length [Default: %default]")
parser.add_option("--species", dest="species",
                  default="human",
                  help="Species for which to compute correlations. One of human or mouse.")
parser.add_option("--subset", dest="subset",
                  default="valid",
                  help="One of train, valid or test.")
parser.add_option("--state_dict", dest="state_dict",
                  default=True,
                  help="True if the model was saved as a state dictionary and DataParallel.")
parser.add_option("--hu", dest="experiments_human",
                  default=5313, type="int",
                  help="Number of output tracks for human [Default: %default]")
parser.add_option("-m", dest="experiments_mouse",
                  default=1643, type="int",
                  help="Number of output tracks for mouse [Default: %default]")
parser.add_option("--linear_enformer", action="store_true", dest="linear_enformer",
                  default=False, 
                  help="whether to import the enformer archtitecture with linear layers in transformer blocks [Default: %default]")
parser.add_option("--data_parallel", action="store_true", dest="data_parallel",
                  default=False, 
                  help="If Data Parallel was used during training, set this to true [Default: %default]")

(options, args) = parser.parse_args()

if len(args) != 4:
    parser.error("Must provide file name, data directory, checkpoint directory and overlap_tss file")
else:
    file = args[0]
    data_dir = args[1]
    checkpoint_dir = args[2]
    overlap_file = args[3]
    
# specify file-name of output gene dictionary
file_name = f"{file}_{overlap_file}_{options.subset}_{options.species}"

print(f"seq-len: {options.seq_length}, overlap_file: {overlap_file}, species: {options.species}, subset: {options.subset}, file_name: {file_name}")

# set torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")


# Read in Genomes
human_fasta_path = f"{data_dir}hg38.ml.fa"
mouse_fasta_path = f"{data_dir}mm10.ml.fa"

# load the dataset
ds = EnformerDataset(options.species, options.subset, 196_608, data_dir, human_fasta_path, mouse_fasta_path, random=False)

if options.linear_enformer:
    from architecture_linear import * 
else:
    from architecture_nolinear import * 

### LOAD MODEL
if options.state_dict:
    m = Enformer.from_hparams(
            dim = 1536,
            depth = 5,
            heads = 8,
            use_checkpointing=True,
            output_heads = dict(human = experiments_human, mouse= experiments_mouse),
            target_length = 896,
        )
    # this line is necessary to be able to load the model!
    if options.data_parallel:
        m = nn.DataParallel(m)  
        m.load_state_dict(torch.load(os.path.join(data_dir, f"{file}.pt")))
        m.to(device)
    else:
        m.load_state_dict(torch.load(os.path.join(data_dir, f"{file}.pt")))
        m.to(device)
                      
else:
    m = torch.load(f"{data_dir}{file}.pt")


# read in the overlap dataframe
overlap_tss = get_overlap(os.path.join(data_dir, f"{options.species}_overlap_tss"), overlap_file)
    
assert np.all(overlap_tss.region_index.isin(ds.input_sequence.region_df["index"]))
by_region = overlap_tss.groupby("region_index")

# initialize an empty dictionary
# for each gene (key) I save a vector of lenght #tracks which contains the predictions/targets for each gene respectively
gene_dict = {"pred": {gene: np.zeros(5313) for gene in overlap_tss.gene_id.unique()},
              "tar": {gene: np.zeros(5313) for gene in overlap_tss.gene_id.unique()}}

print(f"Size of inptu regions dataframe: {ds.input_sequence.region_df.shape}")





# loop over all input sequences of the test set
for i, index in enumerate(ds.input_sequence.region_df["index"]):
    # we are only interested in an input sequence, if it contains at least one gene TSS, so if it is in the dataframe
    if overlap_tss[overlap_tss.region_index == index].shape[0] >= 1:
    #if index in overlap_tss.region_index.values:# == True:
        # get sequence  and target 
        seq, tar = ds.__getitem__(i) # get numpy arrays
        seq = torch.from_numpy(np.expand_dims(seq, axis=0))

        # make prediction with model
        m.eval()
        with torch.no_grad():
            pred = m(seq.to(device))[options.species]
        tar, pred = np.expand_dims(tar,axis=0), pred.detach().cpu().numpy()

        update_gene_dict(single_region_df=by_region.get_group(index), gene_dict=gene_dict, target=tar, prediction=pred)
    else:
        continue



print("Saving gene dictionary")

with open(os.path.join(checkpoint_dir, f"{file_name}_gene_dict.pkl"), "wb") as f:
    pickle.dump(gene_dict, f)
