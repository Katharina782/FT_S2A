import matplotlib.pyplot as plt
import seaborn as sns


from optparse import OptionParser

import math

import torch.optim as optim
import itertools as it
from einops import rearrange

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from architecture_batchNorm_momentum import *
from finetuning_architecture import * 
from data_augmentation import *
from data_utils_finetuning import *

import pickle
import gc

#import tensorflow as tf

# Fine-tune Basenji2 on RNA & ATAC-seq, using a weight for the RNA loss

usage = "usage: %prog [options] <file_name> <data_dir> <checkpoint_dir> <pretrained_model>"
parser = OptionParser(usage)
parser.add_option("-l", dest="seq_length",
                  default=131_072, type="int",
                  help="Input sequence length [Default: %default]")
parser.add_option("--exp_atac", dest="exp_atac",
                  default=None, type="int",
                  help="Number of output tracks for ATAC-seq [Default: %default]")
parser.add_option("--exp_rna", dest="exp_rna",
                  default=None, type="int",
                  help="Number of output tracks for RNA-seq [Default: %default]")
parser.add_option("--log_rna",action="store_true", dest="log_transform_rna",
                  default=False, 
                  help="Log-transform the library size normalized RNA counts [Default: %default]")
parser.add_option("-e", dest="num_epochs",
                  default=200, type="int",
                  help="Number of training epochs [Default: %default]")
parser.add_option("-b", dest="batch_size",
                  default=4, type="int",
                  help="Batch size for training [Default: %default]")
parser.add_option("--shuffle", action="store_true", dest="shuffle",
                  default=False,
                  help="Shuffle data during training")
parser.add_option("--test_data", action="store_true", dest="test_data",
                  default=False,
                  help="Use a small test data set")
parser.add_option("--random_data", action="store_true", dest="random_data",
                  default=False,
                  help="Use random_data set to create a baseline")
parser.add_option("--lr", dest="learning_rate",
                  default=0.15, type="float",
                  help="Learning rate for training [Default: %default]")
parser.add_option("--optimizer", dest="optimizer",
                  default="sgd", type="string",
                  help="Sgd or ADAM [Default: %default]")
parser.add_option("--momentum", dest="momentum",
                  default="0.99", type="float",
                  help="Momentum for optimization [Default: %default]")          
parser.add_option("--dil_mult", dest="dilation_rate_mult",
                  default="1.5", type="float",
                  help="Factor of dilation rate increase [Default: %default]")     
parser.add_option("--bn_momentum", dest="bn_momentum",
                  default="0.99", type="float",
                  help="Batch Norm momentum [Default: %default]")     
parser.add_option("--augment", action="store_true", dest="augment",
                  default=False, 
                  help="Data augmentation [Default: %default]")     
parser.add_option("--rna_weight", dest="rna_weight",
                  default=None, type="float",
                  help="Upweight RNA loss [Default: %default]")    
parser.add_option("--checkpoints", dest="checkpoints",
                  default=2500, type="int",
                  help="Frequency of checkpoints to save loss and models [Default: %default]")
(options, args) = parser.parse_args()


if len(args) != 4:
    parser.error("Must provide file name, data directory and checkpoint directory")
else:
    file_name=args[0]
    data_dir = args[1]
    checkpoint_dir = args[2]
    pretrained_model = args[3]

file_name = f"{file_name}_{options.learning_rate}_{options.optimizer}_{options.batch_size}_augment{options.augment}_atac{options.exp_atac}_rna{options.exp_rna}_batch{options.batch_size}_logtransform{options.log_transform_rna}_loss_weight{options.rna_weight}"

if (options.exp_rna != None) & (options.exp_atac != None):
    experimental_tracks = options.exp_rna + options.exp_atac
    rna = True
    atac = True
elif options.exp_rna != None:
    experimental_tracks = options.exp_rna
    rna = True
    atac = False
elif options.exp_atac != None:
    experimental_tracks = options.exp_atac
    rna = False
    atac = True


print(checkpoint_dir, data_dir, file_name)

print(f"Pretrained model : {pretrained_model}, experiments: atac = {options.exp_atac} rna = {options.exp_rna}, Learning rate: {options.learning_rate}, seq-len: {options.seq_length}, randomized_data: {options.random_data}, test_data: {options.test_data}, batch size: {options.batch_size}, num_epochs: {options.num_epochs}, file_name: {file_name}, shuffle: {options.shuffle}")

# Config

num_dilated_conv = 11
num_conv = 6
conv_target_channels = 768
dilation_rate_init = 1


# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")

human_fasta_path, mouse_fasta_path = os.path.join(data_dir, "hg38.ml.fa"), os.path.join(data_dir, "mm10.ml.fa")


model = FineTuning(data_dir,
                            model_name=pretrained_model, 
                            tracks = experimental_tracks,                
                            num_conv=num_conv,
                            num_dilated_conv=num_dilated_conv,
                            conv_target_channels=conv_target_channels, 
                            bn_momentum=options.bn_momentum, 
                            dilation_rate_init=dilation_rate_init, 
                            dilation_rate_mult=options.dilation_rate_mult, 
                            experiments_human=5313, 
                            experiments_mouse=1643).to(device)


# the old output heads are not updated -> set to not require grad
for name, param in model.named_parameters():
    if "output_heads" in name:
        print(name)
        param.requires_grad = False


if options.test_data: 
    print("Using test data")
    def test_dataloader(seq_length, cage_tracks, atac_tracks, batch_size, shuffle):
        train_data = {}
        for size, name in zip([200, 100], ["train", "valid"]):
            hts, mts = create_random_data(data_size=size, experiments_human=atac_tracks, experiments_mouse=cage_tracks, seq_len=seq_length)
            train_data[f"{name}_human"] = DataLoader(hts, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            train_data[f"{name}_mouse"] = DataLoader(mts, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return train_data

    dataloader = test_dataloader(options.seq_length,
                                 options.experiments_atac,
                                 options.experiments_cage, 
                                 options.batch_size, 
                                 options.shuffle)
    

elif options.random_data == True:
    print("Using random data")
    dataloader = create_torch_dataloader_cross_species(options.seq_length, 
                                                       atac = atac, 
                                                       rna = rna,
                                                       data_dir=data_dir,
                                                        human_fasta_path=os.path.join(data_dir, "hg38.ml.fa"),
                                                         mouse_fasta_path= os.path.join(data_dir, "mm10.ml.fa"), 
                                                        batch_size=options.batch_size, 
                                                        shuffle=options.shuffle, 
                                                        random=True)

else:
    print("Using real data")
    dataloader = create_dataloader_gastr_norm(options.seq_length,
                                                        atac = atac, 
                                                        rna = rna,
                                                        data_dir=data_dir,
                                                        human_fasta_path=os.path.join(data_dir, "hg38.ml.fa"),
                                                        mouse_fasta_path= os.path.join(data_dir, "mm10.ml.fa"),
                                                        rna_data="basenji2_gastrulation_dataset_rna_grcm3_final",  
                                                        log_transform_rna=options.log_transform_rna,
                                                        batch_size=options.batch_size, 
                                                        shuffle=options.shuffle, 
                                                        random=False, 
                                                        distributed_sampler=False)



if options.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)

elif options.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=options.learning_rate, momentum=options.momentum)


if options.augment == True:
    print(f"Data_augmentation")
    stochastic_shift = StochasticShift(shift_max=3, pad_value=0)
    switch_reverse = SwitchReverse()





# keep track of the updates/steps the model makes during training
global_step = 0
train_loss, test_loss = {"mouse":[]}, {"mouse":[]}
separate_loss_train, separate_loss_test = {"rna":[], "atac":[]}, {"rna":[], "atac":[]}

# initialize list to save correlation values
corr_val_list = []
best_loss, best_test_loss = 1e10, 1e10
#poiss = torch.nn.PoissonNLLLoss(log_input=False, reduction="mean")

for epoch in range(options.num_epochs):
    model.train()
    print(f"Starting Epoch {epoch}")

    train_loss_epoch = []
    train_loss_atac_epoch, train_loss_rna_epoch = [], []
    for i, batch in enumerate(dataloader["train"]):
        optimizer.zero_grad()    
        global_step +=1

        # load data
        if options.augment == True:
            sequence, reverse_bool = stochastic_reverse_complement(batch[0], training=True)
            sequence = stochastic_shift.call(sequence, training=True)
            target = switch_reverse.call(([batch[1], reverse_bool]))
            sequence = sequence.to(device)
            target = target.to(device)

        else:
            sequence = batch[0].to(device)
            target = batch[1].to(device)

        out = model(sequence)


        if (rna == True) & (atac == True):
            loss_rna = poisson_loss(out[:, :, -options.exp_rna:], target[:, :, -options.exp_rna:], reduce="mean")
            loss_atac = poisson_loss(out[:, :, :options.exp_atac], target[:, :, :options.exp_atac], reduce="mean")
            print(f"rna loss input:{out[:, :, -options.exp_rna:].shape}, target loss input: {target[:, :, -options.exp_rna:].shape}")
            print(f"ata loss input:{out[:, :, :options.exp_atac].shape}, target loss input: {target[:, :, :options.exp_atac].shape}")
            if options.rna_weight != None:
                loss = (loss_rna * options.rna_weight + loss_atac) / 2
            else: 
                loss = (loss_rna + loss_atac) / 2
            print(f"rna train loss: {loss_rna.item()}, atac train loss: {loss_atac.item()}, out.shape: {out.shape}")
            train_loss_rna_epoch.append(loss_rna.item())
            train_loss_atac_epoch.append(loss_atac.item())
            print(f"total_loss: {loss.item()}")

        elif (rna == False) & (atac == True):
            loss = poisson_loss(out, target, reduce="mean")

        elif (rna == True) & (atac == False):
            loss = poisson_loss(out, target, reduce="mean")

        loss.backward()

        # save loss
        train_loss_epoch.append(loss.item())

        optimizer.step()

        if global_step % 500 == 0:
             
            # append the train loss for this epoch
            train_loss["mouse"].append(sum(train_loss_epoch) / len(train_loss_epoch))
            if (rna == True) & (atac == True):
                separate_loss_train["rna"].append(sum(train_loss_rna_epoch) / len(train_loss_rna_epoch))
                separate_loss_train["atac"].append(sum(train_loss_atac_epoch) / len(train_loss_atac_epoch))


            print(f"Best loss: {best_loss}")
            print(f"New loss: {train_loss['mouse'][-1]}")
            if train_loss["mouse"][-1] <= best_loss:
                best_loss = train_loss["mouse"][-1]
                print("Saving best training model checkpoint")
                save_model = os.path.join(checkpoint_dir, f"{file_name}_model_training_checkpoint.pt")
                torch.save(model.state_dict(), save_model)

            # TESTING
            model.eval()
            test_loss_epoch = []
            test_loss_rna_epoch, test_loss_atac_epoch = [], []
            with torch.no_grad():
                for i, batch in enumerate(dataloader["valid"]):
                        # load data
                        if options.augment == True:
                            sequence, reverse_bool = stochastic_reverse_complement(batch[0], training=True)
                            sequence = stochastic_shift.call(sequence, training=True)
                            target = switch_reverse.call(([batch[1], reverse_bool]))
                            sequence = sequence.to(device)
                            target = target.to(device)

                        else:
                            sequence = batch[0].to(device)
                            target = batch[1].to(device)

                        out = model(sequence)

                        # compute loss
                        if (rna == True) & (atac == True):
                            loss_rna = poisson_loss(out[:, :, -options.exp_rna:], target[:, :, -options.exp_rna:], reduce="mean")
                            loss_atac = poisson_loss(out[:, :, :options.exp_atac], target[:, :, :options.exp_atac], reduce="mean")
                            print(f"rna test loss: {loss_rna.item()}, atac test loss: {loss_atac.item()}, target.shape: {target.shape}")
                            if options.rna_weight != None:
                                loss = (loss_rna * options.rna_weight + loss_atac) /2
                            else:
                                loss = (loss_rna + loss_atac) / 2
                            test_loss_rna_epoch.append(loss_rna.item())
                            test_loss_atac_epoch.append(loss_atac.item())
                            print(f"total_loss: {loss.item()}")

                        elif (rna == False) & (atac == True):
                            loss = poisson_loss(out, target, reduce="mean")
                            
                        elif (rna == True) & (atac == False):
                            loss = poisson_loss(out, target, reduce="mean")
                        test_loss_epoch.append(loss.item())


                            

                print("Computing Correlation")         
                corr_val = compute_correlation_basenji_norm(model, device, data_dir, options.seq_length, atac=atac, rna=rna,log_transform_rna=options.log_transform_rna,
                                            human_fasta_path=human_fasta_path, mouse_fasta_path=mouse_fasta_path, batch_size=options.batch_size, rna_data="basenji2_gastrulation_dataset_rna_grcm3_final",
                                            species="mouse", subset="valid", n_channels=experimental_tracks, shuffle=True, max_steps=400)
                corr_val_list.append(corr_val)
                print(f"current correlation: {corr_val}")
                print("Saving the correlations")                
                with open(f"{checkpoint_dir}{file_name}_correlations.pkl", "wb") as f:
                    pickle.dump(corr_val_list, f)                    

              

                            

            print(f"Testing for epoch {epoch}, step {global_step} finished")
            test_loss["mouse"].append(sum(test_loss_epoch) / len(test_loss_epoch))

            if (rna == True) & (atac == True):
                separate_loss_test["rna"].append(sum(test_loss_rna_epoch) / len(test_loss_rna_epoch))
                separate_loss_test["atac"].append(sum(test_loss_atac_epoch) / len(test_loss_atac_epoch))

            print(f"Best test loss: {best_test_loss}")
            print(f"New test loss: {test_loss['mouse'][-1]}")
            if test_loss["mouse"][-1] <= best_test_loss:
                best_test_loss = test_loss["mouse"][-1]
                print("Saving best validation model checkpoint")
                save_model = os.path.join(checkpoint_dir, f"{file_name}_model_validation_checkpoint.pt")
                torch.save(model.state_dict(), save_model)

            # save the loss files
            print("Saving loss files")
            for file, loss in zip(["train_loss", "test_loss"],[train_loss, test_loss]):
                with open(f"{checkpoint_dir}{file_name}_{file}.pkl", "wb") as f:
                    pickle.dump(loss, f)


            if (rna == True) & (atac == True):
                for file, loss in zip(["train_loss_separate", "test_loss_separate"],[separate_loss_train, separate_loss_test]):
                    with open(f"{checkpoint_dir}{file_name}_{file}.pkl", "wb") as f:
                        pickle.dump(loss, f)