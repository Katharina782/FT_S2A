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
from basenji_architecture_res import * 
#from basenji_architecture import * 
#from original_data_preprocessing import *
from metrics import *
from data_utils import *
from evaluate import * 

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
parser.add_option("--hu", dest="experiments_human",
                  default=5313, type="int",
                  help="Number of output tracks for human [Default: %default]")
parser.add_option("-m", dest="experiments_mouse",
                  default=1643, type="int",
                  help="Number of output tracks for mouse [Default: %default]")

(options, args) = parser.parse_args()

if len(args) != 3:
    parser.error("Must provide file name, data directory, checkpoint directory and overlap_tss file")
else:
    file = args[0]
    data_dir = args[1]
    checkpoint_dir = args[2]


print(f"seq-len: {options.seq_length}, species: {options.species}, file_name: {file}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# model parameters 
num_dilated_conv = 11
num_conv = 6
conv_target_channels = 768
dilation_rate_init = 1

# Read in Genomes
human_fasta_path = f"{data_dir}hg38.ml.fa"
mouse_fasta_path = f"{data_dir}mm10.ml.fa"

### LOAD MODEL
if options.state_dict:
    print("load from torch.state_dict")
    model = BasenjiModel(num_conv, 
                         num_dilated_conv, 
                         conv_target_channels, 
                         bn_momentum=options.bn_momentum,
                         dilation_rate_init=dilation_rate_init, 
                         dilation_rate_mult=options.dilation_rate_mult, 
                         human_tracks=options.experiments_human,
                         mouse_tracks=options.experiments_mouse)   
    model.load_state_dict(torch.load(os.path.join(data_dir, f"{file}_model_validation_checkpoint.pt")))#, map_location=torch.device(device)))
    model.to(device)
else:
    model = torch.load(f"{data_dir}{file}.pt")


# save the gene dictionary for every subset!
for subset in ["train", "test", "valid"]:
    overlap_file = f"overlap_{subset}_protein_coding"

    #create dataset for a subset
    ds = EnformerDataset(options.species, 
                        subset,
                        options.seq_length,
                        data_dir, 
                        human_fasta_path, 
                        mouse_fasta_path, 
                        random=False)

    overlap_tss = get_overlap(os.path.join(data_dir, f"{options.species}_overlap_tss"), overlap_file)
    assert np.all(overlap_tss.region_index.isin(ds.input_sequence.region_df["index"]))
    by_region = overlap_tss.groupby("region_index")


    # initialize an empty dictionary
    # for each gene (key) I save a vector of lenght #tracks which contains the predictions/targets for each gene respectively
    gene_dict = {"pred": {gene: np.zeros(5313) for gene in overlap_tss.gene_id.unique()},
                "tar": {gene: np.zeros(5313) for gene in overlap_tss.gene_id.unique()}}


    # loop over all input sequences of the test set
    for i, index in enumerate(ds.input_sequence.region_df["index"]):
        # we are only interested in an input sequence, if it contains at least one gene TSS, so if it is in the dataframe
        if overlap_tss[overlap_tss.region_index == index].shape[0] >= 1:
        #if index in overlap_tss.region_index.values:# == True:
            # get sequence  and target 
            seq, tar = ds.__getitem__(i) # get numpy arrays
            seq = torch.from_numpy(np.expand_dims(seq, axis=0))

            # make prediction with model
            model.eval()
            with torch.no_grad():
                pred = model(seq.to(device), options.species)
            tar, pred = np.expand_dims(tar,axis=0), pred.detach().cpu().numpy()

            # add the counts for all TSS in the current input region to the gene dictionary
            update_gene_dict(single_region_df=by_region.get_group(index), gene_dict=gene_dict, target=tar, prediction=pred)
        else:
            continue

    print("Saving gene dictionary")

    # specify file-name of output gene dictionary
    file_name = f"{file}_{overlap_file}_{subset}_{options.species}"

    with open(os.path.join(checkpoint_dir, f"{file_name}_gene_dict.pkl"), "wb") as f:
        pickle.dump(gene_dict, f)
        



