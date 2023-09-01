import torch
import numpy as np
#import tensorflow as tf
import os 
import json
import pandas as pd
import pyfaidx
import kipoiseq
import functools
import random
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
experiments_human = 5313
experiments_mouse = 1643



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

        return self.fasta.close()
    



class BasenjiDataSet(torch.utils.data.IterableDataset):
  @staticmethod
  def get_organism_path(organism):
    return os.path.join('/omics/groups/OE0540/internal/users/mikulik/master_thesis/data/gcs_basenj/', organism)
  @classmethod
  def get_metadata(cls, organism):
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join(cls.get_organism_path(organism), 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
      return json.load(f)
  @staticmethod
  def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

  @classmethod
  def get_tfrecord_files(cls, organism, subset):
    # Sort the values by int(*).
    return sorted(tf.io.gfile.glob(os.path.join(
        cls.get_organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
      )), key=lambda x: int(x.split('-')[-1].split('.')[0]))
  
  @property
  def num_channels(self):
    metadata = self.get_metadata(self.organism)
    return metadata['num_targets']

  @staticmethod
  def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile."""
    # Deserialization
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),  # Ignore this, resize our own bigger one
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence_old': sequence,
            'target': target}

  @classmethod
  def get_dataset(cls, organism, subset, num_threads=8):
    metadata = cls.get_metadata(organism)
    dataset = tf.data.TFRecordDataset(cls.get_tfrecord_files(organism, subset),
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_threads).map(
                                          functools.partial(cls.deserialize, metadata=metadata)
                                      )
    return dataset

  def __init__(self, organism:str, subset:str, seq_len:int, fasta_path:str, n_to_test:int = -1):
    assert subset in {"train", "valid", "test"}
    assert organism in {"human", "mouse"}
    self.organism = organism
    self.subset = subset
    self.base_dir = self.get_organism_path(organism)
    self.seq_len = seq_len
    self.fasta_reader = FastaStringExtractor(fasta_path)
    self.n_to_test = n_to_test
    with tf.io.gfile.GFile(f"{self.base_dir}/sequences.bed", 'r') as f:
      region_df = pd.read_csv(f, sep="\t", header=None)
      region_df.columns = ['chrom', 'start', 'end', 'subset']
      self.region_df = region_df.query('subset==@subset').reset_index(drop=True)
      
  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info is None, "Only support single process loading"
    # If num_threads > 1, the following will actually shuffle the inputs! luckily we catch this with the sequence comparison
    basenji_iterator = self.get_dataset(self.organism, self.subset, num_threads=1).as_numpy_iterator()
    for i, records in enumerate(basenji_iterator):
      loc_row = self.region_df.iloc[i]
      target_interval = Interval(loc_row['chrom'], loc_row['start'], loc_row['end'])
      sequence_one_hot = self.one_hot_encode(self.fasta_reader.extract(target_interval.resize(self.seq_len)))
      if self.n_to_test >= 0 and i < self.n_to_test:
        old_sequence_onehot = records["sequence_old"]
        if old_sequence_onehot.shape[0] > sequence_one_hot.shape[0]:
          diff = old_sequence_onehot.shape[0] - sequence_one_hot.shape[0]
          trim = diff//2
          np.testing.assert_equal(old_sequence_onehot[trim:(-trim)], sequence_one_hot)
        elif sequence_one_hot.shape[0] > old_sequence_onehot.shape[0]:
          diff = sequence_one_hot.shape[0] - old_sequence_onehot.shape[0]
          trim = diff//2
          np.testing.assert_equal(old_sequence_onehot, sequence_one_hot[trim:(-trim)])
        else:
          np.testing.assert_equal(old_sequence_onehot, sequence_one_hot)
      yield {
          "sequence": sequence_one_hot,
          "target": records["target"],
      }
     
def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

        
    
class InputSequence():
    def __init__(self, species, subset, seq_length, data_dir, human_fasta_path, mouse_fasta_path, random=False):#, subsample=None):
        '''
        load the region dataframe for a specific subset (train/test/valid) and species (mouse/human)
        If you select random=True, then the indices of the regions will be shuffled to create a random set for training. 
        '''
        file = os.path.join(data_dir, species, "sequences.bed")
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

    def one_hot_encode(self, seq):
        return kipoiseq.transforms.functional.one_hot_dna(seq).astype(np.float32)

        
class EnformerDataset(Dataset):
    def __init__(self, species, subset,seq_length, data_dir, human_fasta_path, mouse_fasta_path, random=False):
        '''
        This class can be used to create a Dataset for training of Enformer or Basenji2 with ENCODE data.

        species (string): one of either"human" or "mouse"
        subset (string): one of either "train", "valid" or "test"
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
        #seqs = InputSequence("human", "train",data_dir, human_fasta_path, subsample = 6)
        
        self.input_sequence = InputSequence(self.species, self.subset, self.seq_length, self.data_dir, 
                                            human_fasta_path, mouse_fasta_path, random=random)
        instances = os.listdir(os.path.join(data_dir, "basenji2_dataset_torch"))#.sort()
        self.instances = [i for i in instances if (self.species in i) & (self.subset in i)]
        assert len(self.input_sequence.region_df) == len(self.instances)
    
    def __getitem__(self, n):
        file = os.path.join(self.data_dir, "basenji2_dataset_torch", f"{self.species}_{self.subset}_{n}.npy")
        self.target = np.load(file)
        self.sequence = self.input_sequence.create_sequence(n)
        assert self.sequence.shape[0] == self.seq_length
        return self.sequence.astype("float32"), self.target.squeeze().astype("float32")
    
    def __len__(self):
        return len(self.instances)
    
    def get_metadata(self):
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
        #file = os.path.join(self.data_dir, self.species, 'statistics.json')
        f = open(os.path.join(self.data_dir, self.species, 'statistics.json'))
        self.metadata = json.load(f)
        return self.metadata
    
    
    def __repr__(self):
        return f"{type(self).__name__}\nSet: {self.subset}\nSpecies: {self.species}\nNum_targets: {self.get_metadata()['num_targets']}\nsequence_length:{self.seq_length}"
    


def create_torch_dataloader_cross_species(seq_length, data_dir, human_fasta_path, mouse_fasta_path,
                                        batch_size, shuffle=False, random=False, distributed_sampler=False):
    '''
    Create a dataloader for the mouse and human species, respectively, stratified by subset (train/test/valid)

    seq_length (int): length of input DNA sequence
    data_dir (string): path to directory of input data
    human_fasta_path (string): path to directory of human reference genome fasta file
    mouse_fasta_path (string): path to directory of mouse reference genome fasta file
    batch_size (int): number of trainings instances used for one update during model training 
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
        for species in ["human", "mouse"]:
            ds = EnformerDataset(species, subset, seq_length, data_dir, human_fasta_path, mouse_fasta_path, random=random)
            #if distributed_sampler == True:
            #    dl_dict["%s_%s" % (subset, species)] = DataLoader(ds,
            #                                                    batch_size=batch_size,
            #                                                    shuffle = shuffle, 
            #                                                    num_workers=0,
            #                                                    drop_last=True,
            #                                                    pin_memory=True,
            #                                                    sampler = DistributedSampler(ds))
            #else: 
            dl_dict["%s_%s" % (subset, species)] = DataLoader(ds,
                                            batch_size=batch_size,
                                            shuffle = shuffle, 
                                            num_workers=0,
                                            drop_last=True,
                                            pin_memory=True)
    return dl_dict

def create_dataset_dictionary(seq_length, data_dir, human_fasta_path, mouse_fasta_path, random=False):
    '''
    Create a dictionary that contains the DataSet objects with  mouse and human species, stratified by subset (train/test/valid) as keys.


    seq_length (int): length of input DNA sequence
    data_dir (string): path to directory of input data
    human_fasta_path (string): path to directory of human reference genome fasta file
    mouse_fasta_path (string): path to directory of mouse reference genome fasta file
    random (bool): If random is set to True, then inside the InputSequence() function, 
                    the indices of the input_sequences will be shuffled, meaning that when we load a target tensor,
                    the corresponding input sequence will be random. 
    '''
    dl_dict = {}        
    for subset in ["train", "valid"]:
        for species in ["human", "mouse"]:
            ds = EnformerDataset(species, subset, seq_length, data_dir, human_fasta_path, mouse_fasta_path, random=random)
            dl_dict["%s_%s" % (subset, species)] = ds
    return dl_dict



def compute_correlation_torch(model, device, data_dir, seq_length, human_fasta_path, mouse_fasta_path, batch_size=1,
                              species:str="human", subset:str="valid", shuffle=False, max_steps=400, per_track=False):
    '''
    Compute the correlation across all positions within a the input regions of specific data subset (train/test/valid). 
    This function can be used for the Enformer model.

    model (torch.nn.Module): trained model
    device (torch.device): device used for model predictions, one of "cuda" or "cpu"
    data_dir (string): path to directory of input data
    seq_length (int): length of input DNA sequence
    human_fasta_path (string): path to directory of human reference genome fasta file
    mouse_fasta_path (string): path to directory of mouse reference genome fasta file
    batch_size (int): number of trainings instances used for one update during model training 
    species (string): one of either"human" or "mouse"
    subset (string): one of either "train", "valid" or "test"
    n_channels (int): number of output tracks of the model
    shuffle (bool): If True, the data instances are shuffled in each epoch
    max_steps (int): number of input regions to be used for correlation computation
    per_track: If True, return one correlation value for each output track, else return the average correlation across output tracks
    '''
    ds = EnformerDataset(species, subset, seq_length, data_dir, human_fasta_path, mouse_fasta_path)
    metadata = ds.get_metadata()
    total = len(ds) # number of records
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)#, shuffle=False)
    print("loaded data for correlation")
    corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=metadata["num_targets"])
    n_steps = total if max_steps <= 0 else max_steps
    for i, batch in enumerate(tqdm(dl)):
        if max_steps > 0 and i >= max_steps:
            break
        sequence = batch[0].to(device)
        target = batch[1].to(device)
        with torch.no_grad():
            pred = model(sequence)[species]
            corr_coef(preds=pred.cpu(), target=target.cpu())

    if per_track: 
        return corr_coef.compute()
    return corr_coef.compute().mean()



def compute_correlation_basenji(model, device, data_dir, seq_length, human_fasta_path, mouse_fasta_path, batch_size=1,
                              species:str="human", subset:str="valid", shuffle=False, max_steps=400, per_track=False):
    '''
    Compute the correlation across all positions within a the input regions of specific data subset (train/test/valid). 
    This function can be used for the Basenji2 model.

    model (torch.nn.Module): trained model
    device (torch.device): device used for model predictions, one of "cuda" or "cpu"
    data_dir (string): path to directory of input data
    seq_length (int): length of input DNA sequence
    human_fasta_path (string): path to directory of human reference genome fasta file
    mouse_fasta_path (string): path to directory of mouse reference genome fasta file
    batch_size (int): number of trainings instances used for one update during model training 
    species (string): one of either"human" or "mouse"
    subset (string): one of either "train", "valid" or "test"
    n_channels (int): number of output tracks of the model
    shuffle (bool): If True, the data instances are shuffled in each epoch
    max_steps (int): number of input regions to be used for correlation computation
    per_track: If True, return one correlation value for each output track, else return the average correlation across output tracks
    '''
    ds = EnformerDataset(species, subset, seq_length, data_dir, human_fasta_path, mouse_fasta_path)
    metadata = ds.get_metadata()
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




# Random dataset for testing
class TS(Dataset):
  def __init__(self, dataset):
    self.dataset = dataset

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    sample = self.dataset[idx]
    return sample

def get_random_input(target_seq_len, experiments, seq_len):
  seq = ''.join(
      [random.choice('ACGT') for _ in range(seq_len)])
  target = np.random.rand(target_seq_len, experiments)
  return kipoiseq.transforms.functional.one_hot_dna(seq).astype(np.float32), target
  #return {"sequence":kipoiseq.transforms.functional.one_hot_dna(seq).astype(np.float32), "target":target}
  #np.expand_dims(enformer.one_hot_encode(seq), 0).astype(np.float32)
    
    
def create_random_data(data_size=100, target_len=896, experiments_human=100, experiments_mouse=60, seq_len=196_608):
  human_ts, mouse_ts = [], []
  for i in range(data_size):
    human_ts.append(get_random_input(target_len, experiments_human, seq_len))
    mouse_ts.append(get_random_input(target_len, experiments_mouse, seq_len))
  human_ds = TS(human_ts)
  mouse_ds = TS(mouse_ts)
  return human_ds, mouse_ds
        
        
# create a data loader dictionary
#def create_dataloader_cross_species(SEQUENCE_LENGTH, human_fasta_path, mouse_fasta_path, batch_size):
#    dl_dict = {}        
#    for subset in ["train", "valid"]:
#        for species in ["human", "mouse"]:
#            fasta_path = human_fasta_path if species == "human" else mouse_fasta_path
#            ds = BasenjiDataSet(species, subset, SEQUENCE_LENGTH, fasta_path)
#            dl_dict["%s_%s" % (subset, species)] = torch.utils.data.DataLoader(ds, 
#                                                                               num_workers=0, 
#                                                                               batch_size=batch_size,
#                                                                              pin_memory=True)
#    return dl_dict
#
#def create_dataloader_single_species(species, fasta_path, batch_size):
#    assert (species == "human") | (species == "mouse")
#    dl_dict = {}
#    for subset in ["train", "valid"]:
#        ds = BasenjiDataSet(species, subset, SEQUENCE_LENGTH, fasta_path)
#        dl_dict["%s_%s" % (subset, species)] = torch.utils.data.DataLoader(ds, 
#                                                                           num_workers=0, 
#                                                                           batch_size=batch_size,
#                                                                          pin_memory=True)
#    return dl_dict
    
    
def get_experiment_indices(data_dir, species):
    targets = pd.read_csv(f"{data_dir}targets_{species}.txt", sep="\t")
    targets[["exp", "description"]] = targets.description.str.split(":",1, expand=True)
    cage = targets[targets["exp"] == "CAGE"].index
    dnase = targets[(targets["exp"] == "DNASE") | (targets["exp"] == "DNASE")].index
    chip = targets[targets["exp"] == "CHIP"].index
    return [cage, dnase, chip]


 
    
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
    dl_dict = create_dataloader_cross_species(seq_length, human_fasta_path, mouse_fasta_path, batch_size=1)
    # for each train/test set and mouse/human we save the targets
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
            if count >= 10:
                break
                
                
                
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
    
    
### DEPRECATED?    
#class EnformerContribution:    
#    '''
#    Compute contribution scores for a region of interest.
#    '''
#    def __init__(self, model_path, model_name):
#        # load a trained model
#        self.model = torch.load(f"{model_path}{model_name}", map_location=torch.device("cpu"))
#    
#    def predict_on_batch(self, inputs, species):
#        # make a model prediction
#        predictions = self.model(inputs)[species]
#        return predictions.detach().numpy()
#        
#    def contribution_input_grad(self, input_sequence, species, target_mask):
#        target_mask_mass = torch.sum(target_mask)
#        input_sequence.requires_grad = True
#        output = self.model(input_sequence)[species]
#        prediction = (target_mask * output / target_mask_mass).sum()
#        prediction.backward()
#        input_sequence.requires_grad = False
#        return (input_sequence.squeeze() * input_sequence.grad.squeeze()).sum(-1)
#
#        