import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import torch.optim as optim

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from data import str_to_one_hot, seq_indices_to_one_hot

from config_enformer import EnformerConfig

from transformers import PreTrainedModel

import numpy as np
import kipoiseq
import os
from torchmetrics.regression.pearson import PearsonCorrCoef

from metrics import *
from data_utils import *


from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt
from optparse import OptionParser


usage = "usage: %prog [options] <file_name> <data_dir> <checkpoint_dir>"
parser = OptionParser(usage)
parser.add_option("-l", dest="seq_length",
                  default=196608, type="int",
                  help="Input sequence length [Default: %default]")
parser.add_option("--steps", dest="max_steps",
                  default=400, type="int",
                  help="Number of sequences to use for correlation computation [Default: %default]")
parser.add_option("-b", dest="batch_size",
                  default=1, type="int",
                  help="Batch size for training [Default: %default]")
parser.add_option("--sp", dest="cross_species",
                  default=True, 
                  help="Enable cross species training [Default: %default]")
parser.add_option("--shuffle", action="store_true", dest="shuffle",
                  default=False,
                  help="Shuffle data during training")
parser.add_option("--species", dest="species",
                  default="human",
                  help="Species for which to compute correlations. One of human or mouse.")
parser.add_option("--subset", dest="subset",
                  default="test",
                  help="One of train, valid or test.")
parser.add_option("--per_exp", action="store_true", dest="per_experiment",
                  default=False,
                  help="Save the correlation for each experimental track in the validation/test set separately")
parser.add_option("--model", dest="model",
                  default="basenji",
                  help="One of basenji or Enformer")
parser.add_option("--linear", action="store_true", dest="linear",
                  default=False,
                  help="Enformer with linear layer in transformer block")
parser.add_option("--data_parallel", action="store_true", dest="data_parallel",
                  default=False,
                  help="Enformer from multi-gpu training")
parser.add_option("--max_steps", dest="max_steps",
                  default=1937, type="int",
                  help="Maximum number of steps for which to compute correlation[Default: %default]")


(options, args) = parser.parse_args()

if len(args) != 3:
    parser.error("Must provide file name, output name, data directory and checkpoint directory")
else:
    file = args[0]
    data_dir = args[1]
    checkpoint_dir = args[2]
    
print(checkpoint_dir, data_dir, file)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")


# Read in Genomes
human_fasta_path = f"{data_dir}hg38.ml.fa"
mouse_fasta_path = f"{data_dir}mm10.ml.fa"


def compute_correlation_basenji(model, device, data_dir, seq_length, human_fasta_path, mouse_fasta_path, batch_size=1,
                              species:str="human", subset:str="valid", shuffle=False, max_steps=400, per_track=False):
    ds = EnformerDataset(species, subset, seq_length, data_dir, human_fasta_path, mouse_fasta_path)
    metadata = ds.get_metadata()
    total = len(ds) # number of records
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)#, shuffle=False)
    print("loaded data for correlation")
    corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=metadata["num_targets"])
    model.eval()
    for i, batch in enumerate(tqdm(dl)):
        if max_steps > 0 and i >= max_steps:
            break
        sequence = batch[0].to(device)
        target = batch[1].to(device)
        with torch.no_grad():
            pred = model(sequence, species)
            corr_coef(preds=pred.cpu(), target=target.cpu())

    if per_track: 
        return corr_coef.compute()
    return corr_coef.compute().mean()



if options.model == "basenji":
    # load trained Basenji model for predictions

    #from Basenji2_torch.architecture_batchNorm_momentum import *
    from Basenji2_torch.basenji_architecture_res import *
    num_dilated_conv = 11
    num_conv = 6
    conv_target_channels = 768
    dilation_rate_init = 1
    bn_momentum = .9
    dilation_rate_mult = 1.5
    experiments_human = 5313
    experiments_mouse = 1643

    model = BasenjiModel( 
                        n_conv_layers=num_conv,
                        n_dilated_conv_layers=num_dilated_conv, 
                        conv_target_channels=conv_target_channels,
                        bn_momentum=bn_momentum,
                        dilation_rate_init=dilation_rate_init, 
                        dilation_rate_mult=dilation_rate_mult, 
                        human_tracks=experiments_human, 
                        mouse_tracks=experiments_mouse)
    model.load_state_dict(torch.load(os.path.join(data_dir, f"{file}.pt")))#, map_location=torch.device("cpu")))
    model.to(device)
    model.eval()

    corr_across_position = compute_correlation_basenji(model,
                                                       device, 
                                                       data_dir, 
                                                       options.seq_length, 
                                                       human_fasta_path, 
                                                       mouse_fasta_path, 
                                                       batch_size=options.batch_size,
                                                       species=options.species, 
                                                       subset=options.subset, 
                                                       shuffle=options.shuffle, 
                                                       max_steps=options.max_steps, 
                                                       per_track=options.per_experiment)


elif options.model == "enformer":
    if options.linear:
        from architecture_linear import * 
    else:
        from architecture_nolinear import * 
    experiments_human = 5313
    experiments_mouse = 1643
    model = Enformer.from_hparams(
                dim = 1536,
                depth = 5,
                heads = 8,
                use_checkpointing=True,
                output_heads = dict(human = experiments_human, mouse= experiments_mouse),
                target_length = 896,
            )
    if options.data_parallel:
         model = nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(data_dir, f"{file}.pt"), map_location=torch.device("cpu")))
    model.to(device)
    model.eval()

    corr_across_position = compute_correlation_torch(model, device, data_dir, 
                                                    options.seq_length, 
                                                    human_fasta_path, mouse_fasta_path,
                                                    batch_size=options.batch_size,
                                                    species=options.species,
                                                    subset=options.subset,
                                                    shuffle=options.shuffle, 
                                                    max_steps=options.max_steps, 
                                                    per_track=options.per_experiment)


print("Saving the correlation vector across positions")
with open(os.path.join(checkpoint_dir, f"{file}_{options.species}_{options.subset}_set_corr_across_pos.pkl"), "wb") as f:
            pickle.dump(corr_across_position, f)

