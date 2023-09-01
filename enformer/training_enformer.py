import os
from optparse import OptionParser
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from data import str_to_one_hot, seq_indices_to_one_hot

from config_enformer import EnformerConfig

from transformers import PreTrainedModel

import numpy as np
import kipoiseq

#from architecture import *
from architecture_linear import * 
from metrics import *
from data_utils import *

import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools as it
from data_augmentation import *




#def main():
usage = "usage: %prog [options] <file_name> <data_dir> <checkpoint_dir>"
parser = OptionParser(usage)
parser.add_option("-l", dest="seq_length",
                  default=196608, type="int",
                  help="Input sequence length [Default: %default]")
parser.add_option("-t", dest="target_length",
                  default=896, type="int",
                  help="Lenght of the prediction target [Default: %default]")
parser.add_option("--hu", dest="experiments_human",
                  default=5313, type="int",
                  help="Number of output tracks for human [Default: %default]")
parser.add_option("-m", dest="experiments_mouse",
                  default=1643, type="int",
                  help="Number of output tracks for mouse [Default: %default]")
parser.add_option("-e", dest="num_epochs",
                  default=200, type="int",
                  help="Number of training epochs [Default: %default]")
parser.add_option("-b", dest="batch_size",
                  default=2, type="int",
                  help="Batch size for training [Default: %default]")
parser.add_option("--sp", dest="cross_species",
                  default=True, 
                  help="Enable cross species training [Default: %default]")
parser.add_option("--shuffle",action="store_true", dest="shuffle",
                  default=False,
                  help="Shuffle data during training")
parser.add_option("-r", dest="return_loss_per_bp",
                  default=False,
                  help="Return an average of the loss [Default: %default]")
parser.add_option("-d", dest="test_data",
                  default=False,
                  help="Use a small test data set")
parser.add_option("--lr", dest="learning_rate",
                  default=1e-5, type="float",
                  help="Learning rate for training [Default: %default]")
parser.add_option("--att_depth", dest="attention_depth",
                  default=5, type="int",
                  help="Number of attention layers [Default: %default]")
parser.add_option("--augment", action="store_true", dest="augment",
                  default=False, 
                  help="Data augmentation [Default: %default]")   
parser.add_option("--checkpoints", dest="checkpoints",
                  default=2500, type="int",
                  help="Frequency of checkpoints to save loss and models [Default: %default]")
(options, args) = parser.parse_args()

if len(args) != 3:
    parser.error("Must provide file name, data directory and checkpoint directory")
else:
    file_name=args[0]
    data_dir = args[1]
    checkpoint_dir = args[2]

print(checkpoint_dir, data_dir, file_name)

file_name = f"{file_name}_{options.learning_rate}_{options.batch_size}_augment{options.augment}"


print(f"Learning rate: {options.learning_rate}, seq-len: {options.seq_length}, target length: {options.target_length},  experiments: human {options.experiments_human}, mouse {options.experiments_mouse}, batch size: {options.batch_size}, num_epochs: {options.num_epochs}, file_name: {file_name}, cross-species: {options.cross_species}, return_loss_per_bp: {options.return_loss_per_bp}, shuffle: {options.shuffle}, number of attention layers:{options.attention_depth}, frequency of checkpoints: {options.checkpoints}")


# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")


print("Creating datasets...")


# Read in genome fasta files for both species
human_fasta_path = f"{data_dir}hg38.ml.fa"
mouse_fasta_path = f"{data_dir}mm10.ml.fa"


# load smaller dataset for fast testing of training loop
if options.test_data:

    experiments_human = options.experiments_human
    experiments_mouse = options.experiments_mouse
    print("Initializing small model...")
    model = Enformer.from_hparams(
        dim = 1536,
        depth = options.attention_depth,
        heads = 8,
        output_heads = dict(human = experiments_human, mouse= experiments_mouse),
        target_length = options.target_length,
    )
    
    
    # if there are more than one GPU available use multi-gpu processing
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.to(device)
    
    # create test training data
    hts, mts = create_random_data(data_size=20, experiments_human=experiments_human, experiments_mouse=experiments_mouse, seq_len=options.seq_length)
    h_train = DataLoader(hts, batch_size=options.batch_size, shuffle=True, num_workers=0)
    m_train = DataLoader(mts, batch_size=options.batch_size, shuffle=True, num_workers=0)

    # create test validation data
    hts, mts = create_random_data(data_size=10, experiments_human=experiments_human, experiments_mouse=experiments_mouse, seq_len=options.seq_length)
    h_valid = DataLoader(hts, batch_size=options.batch_size, shuffle=True, num_workers=0)
    m_valid = DataLoader(mts, batch_size=options
                         .batch_size, shuffle=True, num_workers=0)

    print("Done")

    
    
    
# use ENCODE data for training 
else:
    dl_dict = create_torch_dataloader_cross_species(options.seq_length, data_dir,
                                                    human_fasta_path, mouse_fasta_path, 
                                                    options.batch_size, options.shuffle)
    print(dl_dict)
    
    print("Initializing big model...")
    # Initialize model
    model = Enformer.from_hparams(
        dim = 1536,
        depth = options.attention_depth,
        heads = 8,
        output_heads = dict(human = options.experiments_human, mouse= options.experiments_mouse),
        target_length = options.target_length,
    )
    
    #print(f"Model deppth/number of attention layers: {model.depth}")
    
    #print(model)
    
    # if there are more than one GPU available use multi-gpu processing
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(device)

    print("Done")




if options.augment == True:
    print(f"Data_augmentation")
    stochastic_shift = StochasticShift(shift_max=3, pad_value=0)
    switch_reverse = SwitchReverse()


print("Start training...")
print(f"Device: {device}")


#################### TRAINING PREPARATION ##################
optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)
# initalize dictionary to save the trianing and validation loss
train_loss = {"human":[], "mouse":[]}
test_loss = {"human":[], "mouse":[]}
# initialize list to save correlation values
corr_val_list = []
# keep track of the updates/steps the model makes during training
global_step=0
best_valid_loss, best_train_loss = 1e10, 1e10

for epoch in range(options.num_epochs):
    print(f"Starting epoch {epoch}")
    if options.test_data == True:
        hdl, mdl = iter(h_train), iter(m_train)
    else:
        hdl, mdl = dl_dict["train_human"], dl_dict["train_mouse"]

    ################## TRAIN #######################
    train_loss_epoch_human, train_loss_epoch_mouse = [], []
    #for i, batch in enumerate(zip(tqdm(hdl), tqdm(mdl))):
    
    # the cycle() function makes sure that we do not run out of training instances
    # the mouse dataset is smaller than the human datset, so we recycle some of the mouse
    # instances within one epoch
    for i, batch in enumerate(zip(hdl, mdl)): 
        model.train()
        for h, head in enumerate(["human", "mouse"]):
            #print(f"Head:{head}, global_step:{global_step}")
            global_step += 1
            #print(global_step)

            if options.augment == True:
                sequence, reverse_bool = stochastic_reverse_complement(batch[h][0], training=True)
                sequence = stochastic_shift.call(sequence, training=True)
                target = switch_reverse.call(([batch[h][1], reverse_bool]))
                sequence = sequence.to(device)
                target = target.to(device)

            else:
                sequence = batch[h][0].to(device)
                target = batch[h][1].to(device)



            #sequence = batch[h][0].to(device)
            #target = batch[h][1].to(device)
            #print(sequence.shape, target.shape)  
            optimizer.zero_grad()
            loss = model(
              sequence,
              target = target,
              return_corr_coef = False,
              return_embeddings = False,
              return_only_embeddings = False,
              return_loss_per_bp = options.return_loss_per_bp,
              head = head)
            #loss = poisson_loss(pred, target)
            #print(type(loss))
            #print(loss)
            #print(loss.mean().item())
            if head == "human":
                train_loss_epoch_human.append(loss.detach().cpu())#.mean())
            else: 
                train_loss_epoch_mouse.append(loss.detach().cpu())#.mean())
            #loss.sum().backward()
            loss.backward()
            #loss.mean().backward()
            optimizer.step()
            
            ######### CHECKPOINTS #################
            # In the first epochs/the first update steps, the loss decreases rapidly
            # and the correlation increases rapidly, so fewer checkpoints are necessary 
            
            
            ################ STARTING CHECKPOINT FREQUENCY ###################
            if global_step < 25_000:
                #print(f"Global_step: {global_step}, epoch: {epoch}")
                if global_step % (options.checkpoints*3) == 0:
                    model.eval()

                    print("Computing Correlation")
                    corr_val = compute_correlation_torch(model, device, data_dir, options.seq_length, 
                                                         human_fasta_path, mouse_fasta_path, batch_size=options.batch_size,
                                                           species="human", subset="valid", shuffle=True, max_steps=100)
                    #corr_val = compute_correlation(model, device, human_fasta_path, organism="human", subset="test", max_steps=100)
                    corr_val_list.append(corr_val)
                    print(f"Saving the correlations as {file_name}_correlations.pkl")
                    with open(f"{checkpoint_dir}{file_name}_correlations.pkl", "wb") as f:
                        pickle.dump(corr_val_list, f)



                    print("Computing test loss")

                    if options.test_data:                            
                        hdl_test, mdl_test = iter(h_valid), iter(m_valid)
                    else:
                        hdl_test, mdl_test = dl_dict["valid_human"], dl_dict["valid_mouse"]
                        #hdl_test, mdl_test = test_human_loader, test_mouse_loader              

                    test_loss_epoch_mouse, test_loss_epoch_human = [], []
                    with torch.no_grad():
                        for i, batch in enumerate(zip(hdl_test, mdl_test)):
                            for h, head in enumerate(["human", "mouse"]):

                                if options.augment == True:
                                    sequence, reverse_bool = stochastic_reverse_complement(batch[h][0], training=True)
                                    sequence = stochastic_shift.call(sequence, training=True)
                                    target = switch_reverse.call(([batch[h][1], reverse_bool]))
                                    sequence = sequence.to(device)
                                    target = target.to(device)

                                else:
                                    sequence = batch[h][0].to(device)
                                    target = batch[h][1].to(device)
                                #sequence = batch[h][0].to(device)
                                #target = batch[h][1].to(device)
                                loss = model(
                                  sequence,
                                  target = target,
                                  return_corr_coef = False,
                                  return_embeddings = False,
                                  return_only_embeddings = False,
                                  return_loss_per_bp = options.return_loss_per_bp,
                                  head = head)
                                if head == "human":
                                    test_loss_epoch_human.append(loss.detach().cpu())#.mean())
                                else:
                                    test_loss_epoch_mouse.append(loss.detach().cpu())#.mean())
                    print("Testing done")



                    # append the train loss & test loss
                    for loss, species in zip([train_loss_epoch_human, train_loss_epoch_mouse], ["human", "mouse"]):
                        train_loss[species].append(sum(loss) / len(loss))
                    for loss, species in zip([test_loss_epoch_human, test_loss_epoch_mouse], ["human", "mouse"]):
                        test_loss[species].append(sum(loss) / len(loss))

                    # save the loss files
                    for file, loss in zip(["train_loss", "test_loss"],[train_loss, test_loss]):
                        with open(f"{checkpoint_dir}{file_name}_{file}.pkl", "wb") as f:
                            pickle.dump(loss, f)

                    print(f'Current training loss: {train_loss["human"][-1]}, best loss so far: {best_train_loss}')
                    if train_loss["human"][-1] <= best_train_loss:  
                        best_train_loss = train_loss["human"][-1]
                        torch.save(obj=model.state_dict(), f=f"{checkpoint_dir}{file_name}_training_state_dict.pt")

                    print(f'Current validation loss: {train_loss["human"][-1]}, best loss so far: {best_valid_loss}')
                    if test_loss["human"][-1] <= best_valid_loss:  
                        best_train_loss = test_loss["human"][-1]
                        torch.save(obj=model.state_dict(), f=f"{checkpoint_dir}{file_name}_validation_state_dict.pt")


                
            ################ INCREASED CHECKPOINT FREQUENCY ###################
            # increase frequency of checkpoints after 25_000 steps
            else: 
                #print(f"Global_step: {global_step}, epoch: {epoch}")
                # compute correlation every 1000 steps
                if global_step % options.checkpoints == 0:
                    model.eval()

                    print("Computing Correlation")
                    corr_val = compute_correlation_torch(model, device, data_dir, options.seq_length, 
                                                         human_fasta_path, mouse_fasta_path, batch_size=options.batch_size,
                                                         species="human", subset="valid", shuffle=True, max_steps=100)
                    #corr_val = compute_correlation(model, device, human_fasta_path, organism="human", subset="test", max_steps=100)
                    corr_val_list.append(corr_val)
                    print("Saving the correlations")
                    with open(f"{checkpoint_dir}{file_name}_correlations.pkl", "wb") as f:
                        pickle.dump(corr_val_list, f)



                    print("Computing test loss")

                    if options.test_data:                            
                        hdl_test, mdl_test = iter(h_valid), iter(m_valid)
                    else:
                        hdl_test, mdl_test = dl_dict["valid_human"], dl_dict["valid_mouse"]
                        #hdl_test, mdl_test = test_human_loader, test_mouse_loader              

                    test_loss_epoch_mouse, test_loss_epoch_human = [], []
                    with torch.no_grad():
                        for i, batch in enumerate(zip(tqdm(hdl_test), tqdm(it.cycle(mdl_test)))):
                            for h, head in enumerate(["human", "mouse"]):
                                if options.augment == True:
                                    sequence, reverse_bool = stochastic_reverse_complement(batch[h][0], training=True)
                                    sequence = stochastic_shift.call(sequence, training=True)
                                    target = switch_reverse.call(([batch[h][1], reverse_bool]))
                                    sequence = sequence.to(device)
                                    target = target.to(device)

                                else:
                                    sequence = batch[h][0].to(device)
                                    target = batch[h][1].to(device)
                                #sequence = batch[h][0].to(device)
                                #target = batch[h][1].to(device)
                                loss = model(
                                  sequence,
                                  target = target,
                                  return_corr_coef = False,
                                  return_embeddings = False,
                                  return_only_embeddings = False,
                                  return_loss_per_bp = options.return_loss_per_bp,
                                  head = head)
                                if head == "human":
                                    test_loss_epoch_human.append(loss.detach().cpu())#.mean())
                                else:
                                    test_loss_epoch_mouse.append(loss.detach().cpu())#.mean())
                    print("Testing done")



                    # append the train loss
                    for loss, species in zip([train_loss_epoch_human, train_loss_epoch_mouse], ["human", "mouse"]):
                        train_loss[species].append(sum(loss) / len(loss))
                    # append the test loss
                    for loss, species in zip([test_loss_epoch_human, test_loss_epoch_mouse], ["human", "mouse"]):
                        test_loss[species].append(sum(loss) / len(loss))

                    # save the loss files
                    for file, loss in zip(["train_loss", "test_loss"],[train_loss, test_loss]):
                        with open(f"{checkpoint_dir}{file_name}_{file}.pkl", "wb") as f:
                            pickle.dump(loss, f)

                    print(f'Current training loss: {train_loss["human"][-1]}, best loss so far: {best_train_loss}')
                    if train_loss["human"][-1] <= best_train_loss:  
                        best_train_loss = train_loss["human"][-1]
                        torch.save(obj=model.state_dict(), f=f"{checkpoint_dir}{file_name}_training_state_dict.pt")

                    print(f'Current validation loss: {train_loss["human"][-1]}, best loss so far: {best_valid_loss}')
                    if test_loss["human"][-1] <= best_valid_loss:  
                        best_train_loss = test_loss["human"][-1]
                        torch.save(obj=model.state_dict(), f=f"{checkpoint_dir}{file_name}_validation_state_dict.pt")

                    ## get the train loss at the previous checkpoint and compare to current train loss
                    ## here I only check the human loss 
                    ## if the trainings loss has decreased I overwrite the saved training model checkpoint
                    #if len(train_loss["human"]) > 3: # make sure the list is actually long enough
                    #    if train_loss["human"][-1] < train_loss["human"][-2]:                                                                             
                    #        torch.save(obj=model.state_dict(), f=f"{checkpoint_dir}{file_name}_training_state_dict.pt")
                    #        
                    ## get the validation loss at the previous checkpoint and compare it to the current validation loss
                    ## here I only check the human loss
                    ## if the validation loss has decreased I overwrite the saved validation model checkpoint
                    #if len(test_loss["human"]) > 3:
                    #    if test_loss["human"][-1] < test_loss["human"][-2]:
                    #        torch.save(obj=model.state_dict(), f=f"{checkpoint_dir}{file_name}_validation_state_dict.pt")
