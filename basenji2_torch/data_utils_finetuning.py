import torch
import numpy as np
import tensorflow as tf
import os 
import json
import pandas as pd
import pyfaidx
import kipoiseq
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler

from kipoiseq import Interval
from torch.utils.data import Dataset, DataLoader
from metrics import *
from sklearn.utils import shuffle



SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896
BIN_SIZE = 128
experiments = 35



def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream


class InputSequence():
    def __init__(self, species, subset, seq_length, data_dir, human_fasta_path, mouse_fasta_path, random=False):#, subsample=None):
        '''
        Load the region dataframe for a specific subset (train/test/valid) and species (mouse/human)
        If you select random=True, then the indices of the regions will be shuffled to create a random set for training. 
        '''
        file = os.path.join(data_dir, "mouse", "sequences.bed")
        region_df = pd.read_csv(file, sep="\t", header=None)
        region_df.columns = ['chrom', 'start', 'end', 'subset']
        self.seq_length = seq_length
        region_df = region_df.query('subset==@subset').reset_index(drop=False)
        #if not subsample == None:
         #   region_df = region_df[:subsample]
        if random:
            self.region_df = shuffle(region_df)
        else:
            self.region_df = region_df
            
        self.fasta_reader = FastaStringExtractor(human_fasta_path) if species == "human" else FastaStringExtractor(mouse_fasta_path)
        
    def create_sequence(self, index):
        ''' 
        from a certain subset (train/test/valid) and species (human/mouse) extract the input sequence one-hot-encoded
        '''
        seq = self.region_df.iloc[index]
        seq = kipoiseq.Interval(seq.chrom, seq.start, seq.end).resize(self.seq_length)
        return one_hot_encode(self.fasta_reader.extract(seq))




class EnformerDatasetGastrNorm(Dataset):
    def __init__(self, species, subset, atac, rna, seq_length, data_dir, human_fasta_path,
                  mouse_fasta_path, rna_data="basenji2_gastrulation_dataset_rna_grcm3_final", 
                  log_transform_rna=False, random=False):
        '''
        This DataSet class is specific for mouse gastrulation snRNA- and snATAC-seq data. 
        There is the option to log-transform the RNA data.

        species (string): one of either"human" or "mouse"
        subset (string): one of either "train", "valid" or "test"
        atac (bool): If atac is set to True, then the dataset will contain ATAC-seq data target rracks
        rna (bool): If rna is set to True, then the dataset will contain RNA-seq data target tracks
        seq_length (int): length of input DNA sequence
        data_dir (string): path to directory of input data
        rna_data (string): foler within the data_dir that contains library-size normalized RNA data
        log_transform_rna (bool): If log_transform_rna is set to True, then the RNA data will be log-transformed
        human_fasta_path (string): path to directory of human reference genome fasta file
        mouse_fasta_path (string): path to directory of mouse reference genome fasta file
        random (bool): If random is set to True, then inside the InputSequence() function, 
                        the indices of the input_sequences will be shuffled, meaning that when we load a target tensor,
                        the corresponding input sequence will be random. 
        '''
        self.species = species
        self.subset = subset
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.atac, self.rna = atac, rna
        self.log_transform = log_transform_rna
        self.rna_data_path = rna_data
        #seqs = InputSequence("human", "train",data_dir, human_fasta_path, subsample = 6)
        
        self.input_sequence = InputSequence(self.species, self.subset, self.seq_length, self.data_dir, 
                                            human_fasta_path, mouse_fasta_path, random=random)
        instances = os.listdir(os.path.join(data_dir, "basenji2_gastrulation_dataset"))#.sort()
        self.instances = [i for i in instances if (self.subset in i)]
        assert len(self.input_sequence.region_df) == len(self.instances)
    
    def __getitem__(self, n):
        if (self.atac == True) & (self.rna == True):
            file1 = os.path.join(self.data_dir, "basenji2_gastrulation_dataset", f"{self.subset}-{n}.npy")
            file2 = os.path.join(self.data_dir, self.rna_data_path, f"rna_{self.subset}-{n}.npy")
            self.target1 = np.load(file1)
            self.target2 = np.load(file2)
            if self.log_transform:  
                #print("log transform RNA")
                self.target2 = np.log(self.target2 + 1)
            self.target = np.concatenate((self.target1, self.target2), axis=1)
        elif (self.rna == True) & (self.atac == False):
            file2 = os.path.join(self.data_dir, self.rna_data_path, f"rna_{self.subset}-{n}.npy")
            self.target = np.load(file2)
            if self.log_transform:
                #print("log transform RNA")
                self.target = np.log(self.target + 1)
        if (self.rna == False) & (self.atac == True):
            file1 = os.path.join(self.data_dir, "basenji2_gastrulation_dataset", f"{self.subset}-{n}.npy")
            self.target = np.load(file1)
        self.sequence = self.input_sequence.create_sequence(n)
        assert self.sequence.shape[0] == self.seq_length
        return self.sequence.astype("float32"), self.target.squeeze().astype("float32")
    
    def __len__(self):
        return len(self.instances)
    
    def get_metadata(self):
        f = open(os.path.join(self.data_dir, self.species, 'statistics.json'))
        self.metadata = json.load(f)
        self.metadata["num_targets"] = 35
        return self.metadata
    
    
    def __repr__(self):
        return f"{type(self).__name__}\nSet: {self.subset}\nSpecies: {self.species}\nNum_targets: {self.get_metadata()['num_targets']}\nsequence_length:{self.seq_length}"
    


def create_dataloader_gastr_norm(seq_length, atac, rna, data_dir, human_fasta_path, mouse_fasta_path, batch_size, 
                                 rna_data="basenji2_gastrulation_dataset_rna_grcm3_final" ,
                                 log_transform_rna=False, shuffle=False, random=False, distributed_sampler=False):
    '''
    Create a dataloader for the mouse and human species, respectively, stratified by subset (train/test/valid)

    seq_length (int): length of input DNA sequence
    atac (bool): If atac is set to True, then the dataset will contain ATAC-seq data target rracks
    rna (bool): If rna is set to True, then the dataset will contain RNA-seq data target tracks
    data_dir (string): path to directory of input data
    human_fasta_path (string): path to directory of human reference genome fasta file
    mouse_fasta_path (string): path to directory of mouse reference genome fasta file
    batch_size (int): number of trainings instances used for one update during model training 
    rna_data (string): foler within the data_dir that contains library-size normalized RNA data
    log_transform_rna (bool): If log_transform_rna is set to True, then the RNA data will be log-transformed
    shuffle (bool): If True, the data instances are shuffled in each epoch
    random (bool): If random is set to True, then inside the InputSequence() function, 
                    the indices of the input_sequences will be shuffled, meaning that when we load a target tensor,
                    the corresponding input sequence will be random. 
    distributed_sampler (bool): If True, shuffle is set to False, for multi-gpu training with DistributedSampler
    '''
    dl_dict = {}        
    if distributed_sampler:
        assert shuffle == False, "Shuffle must be False if using DistributedSampler!"

    for subset in ["train", "valid", "test"]:
        ds = EnformerDatasetGastrNorm("mouse", subset, atac, rna, seq_length, data_dir, human_fasta_path, mouse_fasta_path, rna_data=rna_data, log_transform_rna=log_transform_rna, random=random)

        dl_dict["%s" % (subset)] = DataLoader(ds,
                                        batch_size=batch_size,
                                        shuffle = shuffle, 
                                        num_workers=0,
                                        drop_last=True,
                                        pin_memory=True)
                                
    return dl_dict
    

    
def compute_correlation_basenji_norm(model, device, data_dir, seq_length, human_fasta_path, mouse_fasta_path, 
                                    atac, rna, rna_data="basenji2_gastrulation_dataset_rna_grcm3_final", 
                                    log_transform_rna=False, batch_size=1, species:str="mouse", subset:str="valid", 
                                    n_channels=35, shuffle=False, max_steps=400, per_track=False):
    '''
    Compute the correlation across all positions within a the input regions of specific data subset (train/test/valid). 
    This function can be used for the Basenji model, fine-tuned on the mouse gastrulation dataset.

    model (torch.nn.Module): trained model
    device (torch.device): device used for model predictions, one of "cuda" or "cpu"
    data_dir (string): path to directory of input data
    seq_length (int): length of input DNA sequence
    human_fasta_path (string): path to directory of human reference genome fasta file
    mouse_fasta_path (string): path to directory of mouse reference genome fasta file
    atac (bool): If atac is set to True, then the dataset will contain ATAC-seq data target rracks
    rna (bool): If rna is set to True, then the dataset will contain RNA-seq data target tracks
    rna_data (string): foler within the data_dir that contains library-size normalized RNA data
    log_transform_rna (bool): If log_transform_rna is set to True, then the RNA data will be log-transformed
    batch_size (int): number of trainings instances used for one update during model training 
    species (string): one of either"human" or "mouse"
    subset (string): one of either "train", "valid" or "test"
    n_channels (int): number of output tracks of the model
    shuffle (bool): If True, the data instances are shuffled in each epoch
    max_steps (int): number of input regions to be used for correlation computation
    per_track: If True, return one correlation value for each output track, else return the average correlation across output tracks
    '''
    # Create a dataset containing the correspoding subset (train/valid/test)
    ds = EnformerDatasetGastrNorm(species, subset, atac, rna, seq_length, data_dir,
                                  human_fasta_path, mouse_fasta_path, rna_data=rna_data, log_transform_rna=log_transform_rna,random=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    print("loaded data for correlation")
    corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=n_channels)

    for i, batch in enumerate(tqdm(dl)):
        if max_steps > 0 and i >= max_steps:
            break
        model.eval()
        sequence = batch[0].to(device)
        target = batch[1].to(device)
        with torch.no_grad():
            if (atac == True) & (rna == True):
                pred = model(sequence)
                pred = pred.cpu()[:, :, :35]
            else:
                pred = model(sequence)
                corr_coef(preds=pred.cpu(), target=target.cpu())

    if per_track: 
        return corr_coef.compute()
    return corr_coef.compute().mean()

def compute_correlation_basenji_scratch(model, device, data_dir, seq_length, 
                                        human_fasta_path, mouse_fasta_path, atac, rna, batch_size=1,
                                        rna_data="basenji2_gastrulation_dataset_rna_grcm3_final",log_transform_rna=False,
                                        species:str="mouse", subset:str="valid", shuffle=False, max_steps=400, per_track=False, n_channels=35):
    '''
    Compute the correlation across all positions within a the input regions of specific data subset (train/test/valid). 
    This function can be used for the Basenji model, trained from scratch on the mouse gastrulation dataset.

    model (torch.nn.Module): trained model
    device (torch.device): device used for model predictions, one of "cuda" or "cpu"
    data_dir (string): path to directory of input data
    seq_length (int): length of input DNA sequence
    human_fasta_path (string): path to directory of human reference genome fasta file
    mouse_fasta_path (string): path to directory of mouse reference genome fasta file
    atac (bool): If atac is set to True, then the dataset will contain ATAC-seq data target rracks
    rna (bool): If rna is set to True, then the dataset will contain RNA-seq data target tracks
    rna_data (string): foler within the data_dir that contains library-size normalized RNA data
    log_transform_rna (bool): If log_transform_rna is set to True, then the RNA data will be log-transformed
    batch_size (int): number of trainings instances used for one update during model training 
    species (string): one of either"human" or "mouse"
    subset (string): one of either "train", "valid" or "test"
    n_channels (int): number of output tracks of the model
    shuffle (bool): If True, the data instances are shuffled in each epoch
    max_steps (int): number of input regions to be used for correlation computation
    per_track: If True, return one correlation value for each output track, else return the average correlation across output tracks
    '''   

    
    ds = EnformerDatasetGastrNorm(species, subset, atac, rna, seq_length, data_dir,
                                   human_fasta_path, mouse_fasta_path,  rna_data=rna_data,
                                     log_transform_rna=log_transform_rna,random=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)#, shuffle=False)
    print("loaded data for correlation")
    corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=n_channels)
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

                
def plot_tracks(tracks, interval, height=1.5):
    '''
    Plot the target tracks or predictions of a specific interval.
    This code is isnpired by https://github.com/deepmind/deepmind-research/tree/master/enformer
    '''
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    



#### OLD function
class EnformerDatasetGastr(Dataset):
    def __init__(self, species, subset, atac, rna, seq_length, data_dir, human_fasta_path, mouse_fasta_path, random=False):
        '''
        This DataSet class is specific for mouse gastrulation snRNA- and snATAC-seq data.

        species (string): one of either"human" or "mouse"
        subset (string): one of either "train", "valid" or "test"
        atac (bool): If atac is set to True, then the dataset will contain ATAC-seq data target rracks
        rna (bool): If rna is set to True, then the dataset will contain RNA-seq data target tracks
        seq_length (int): length of input DNA sequence
        data_dir (string): path to directory of input data
        human_fasta_path (string): path to directory of human reference genome fasta file
        mouse_fasta_path (string): path to directory of mouse reference genome fasta file
        random (bool): If random is set to True, then inside the InputSequence() function, 
                        the indices of the input_sequences will be shuffled, meaning that when we load a target tensor,
                        the corresponding input sequence will be random. 
        '''
        self.species = species
        self.subset = subset
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.atac, self.rna = atac, rna
        # create InputSequence object        
        self.input_sequence = InputSequence(self.species, self.subset, self.seq_length, self.data_dir, 
                                            human_fasta_path, mouse_fasta_path, random=random)
        instances = os.listdir(os.path.join(data_dir, "basenji2_gastrulation_dataset"))
        # select the subset of interest (train/test/valid)
        self.instances = [i for i in instances if (self.subset in i)]
        assert len(self.input_sequence.region_df) == len(self.instances)
    
    def __getitem__(self, n):
        # load the n-th target track 

        # for both RNA and ATAC
        if (self.atac == True) & (self.rna == True):
            file1 = os.path.join(self.data_dir, "basenji2_gastrulation_dataset", f"{self.subset}-{n}.npy")
            file2 = os.path.join(self.data_dir, "basenji2_gastrulation_dataset_rna_grcm3_final", f"rna_{self.subset}-{n}.npy")
            self.target1 = np.load(file1)
            self.target2 = np.load(file2) * 10_000
            self.target = np.concatenate((self.target1, self.target2), axis=1)
        # for just RNA
        elif (self.rna == True) & (self.atac == False):
            file2 = os.path.join(self.data_dir, "basenji2_gastrulation_dataset_rna_grcm3_final", f"rna_{self.subset}-{n}.npy")
            self.target = np.load(file2) * 10_000
        # for just ATAC
        if (self.rna == False) & (self.atac == True):
            file1 = os.path.join(self.data_dir, "basenji2_gastrulation_dataset", f"{self.subset}-{n}.npy")
            self.target = np.load(file1)
        
        # load the n-th input sequence for
        self.sequence = self.input_sequence.create_sequence(n)
        # make sure that the input sequence has the correct shape
        assert self.sequence.shape[0] == self.seq_length

        return self.sequence.astype("float32"), self.target.squeeze().astype("float32")
    
    def __len__(self):
        return len(self.instances)
    
    def get_metadata(self):
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
        #file = os.path.join(self.data_dir, self.species,'statistics.json')
        f = open(os.path.join(self.data_dir, self.species, 'statistics.json'))
        self.metadata = json.load(f)
        self.metadata["num_targets"] = 35
        return self.metadata
    
    
    def __repr__(self):
        return f"{type(self).__name__}\nSet: {self.subset}\nSpecies: {self.species}\nNum_targets: {self.get_metadata()['num_targets']}\nsequence_length:{self.seq_length}"
    