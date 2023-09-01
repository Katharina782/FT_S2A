import os
from optparse import OptionParser
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from data import str_to_one_hot, seq_indices_to_one_hot
    

import tensorflow as tf

import numpy as np
import kipoiseq

#from architecture import *
from data_utils import *

import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import kipoiseq




SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896
BIN_SIZE = 128

data_dir = "/omics/groups/OE0540/internal/users/mikulik/master_thesis/data/gcs_basenj"
human_fasta_path = os.path.join(data_dir, "hg38.ml.fa")
mouse_fasta_path = os.path.join(data_dir, "mm10.ml.fa")

def write_targets_old(data_dir, SEQUENCE_LENGTH, human_fasta_path, mouse_fasta_path, batch_size=1):
    dl_dict = create_dataloader_cross_species(SEQUENCE_LENGTH, human_fasta_path, mouse_fasta_path, batch_size=1)
    for key in dl_dict.keys():
        dl = dl_dict[key]
        subset, species = key.split("_")#[0], key.split()[1]
        save_dir = os.path.join(data_dir, "basenji2_dataset_torch")
        if os.path.isdir(save_dir):
            print("Path exists")
        else:
            print("Creating path")
            os.makedirs(save_dir)
        count = 0
        for i, instance in enumerate(iter(dl)):
            target = instance["target"].numpy()
            np.save(os.path.join(save_dir, "%s_%s_%s.npy" % (species, subset, str(count))), target)
            count+=1
            



# create a data loader dictionary
def create_dataloader_cross_species_test(SEQUENCE_LENGTH, human_fasta_path, mouse_fasta_path, batch_size):
    dl_dict = {}        
    for subset in ["train", "valid"]:
        for species in ["human", "mouse"]:
            fasta_path = human_fasta_path if species == "human" else mouse_fasta_path
            ds = BasenjiDataSet(species, subset, SEQUENCE_LENGTH, fasta_path)
            dl_dict["%s_%s" % (subset, species)] = torch.utils.data.DataLoader(ds, 
                                                                               num_workers=0, 
                                                                               batch_size=batch_size,
                                                                              pin_memory=True)
    return dl_dict
            

def write_targets(data_dir, seq_length, human_fasta_path, mouse_fasta_path, batch_size=1):
    '''
    Write the target tracks for every sequence to a separate npy file.
    data_dir: (string) path to where your data is stored
    seq_length: (int) Length of input DNA sequence
    human_fasta_path: (str) path to the human reference genome fasta file
    mouse_fasta_path: (str) path to the mouse reference genome fasta file
    batch_size: (int) number of data instances to load and save in each file.
    '''
    # load the data as tensorflow dataloader
    dl_dict = create_dataloader_cross_species_test(seq_length, human_fasta_path, mouse_fasta_path, batch_size=1)
    # for each train/test set and mouse/human we save the targets
    for key in dl_dict.keys():
        dl = dl_dict[key]
        subset, species = key.split("_")
        print(subset, species)
        save_dir = os.path.join(data_dir, "basenji2_dataset_torch_new")
        if os.path.isdir(save_dir):
            print("Path exists")
        else:
            print("Creating path")
            os.makedirs(save_dir)
        count = 0
        if species == "human":
            tensor_shape = (1, 896, 5313)
        else:
            tensor_shape = (1, 896, 1643)
        # iterate over all target instances of a ceratin subset/species combination
        for i, instance in enumerate(iter(dl)):
            target = instance["target"].numpy() # save as numpy arrays
            print(target.shape)
            assert target.shape == tensor_shape
            print(target.shape == tensor_shape)
            np.save(os.path.join(save_dir, "%s_%s_%s.npy" % (species, subset, str(count))), target)
            count+=1
            #if count >= 10:
             #   break
write_targets(data_dir, 196_608, human_fasta_path, mouse_fasta_path, batch_size=1)