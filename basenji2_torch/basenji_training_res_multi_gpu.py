import matplotlib.pyplot as plt
import seaborn as sns


from optparse import OptionParser


import torch.optim as optim
import itertools as it
from einops import rearrange

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from basenji_architecture_res import *
from data_augmentation import *
from data_utils import *

import pickle

import tensorflow as tf

## For training Basenji2'', the debugged architecture on several GPUs

# activation after the residual connection

usage = "usage: %prog [options] <file_name> <data_dir> <checkpoint_dir>"
parser = OptionParser(usage)
parser.add_option("-l", dest="seq_length",
                  default=131_072, type="int",
                  help="Input sequence length [Default: %default]")
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
parser.add_option("--shuffle", action="store_true", dest="shuffle",
                  default=False,
                  help="Shuffle data during training")
parser.add_option("--test_data", action="store_true", dest="test_data",
                  default=False,
                  help="Use a small test data set")
parser.add_option("--random_data", action="store_true", dest="random_data",
                  default=False,
                  help="Use random_data set to create a baseline")
parser.add_option("--tensorflow_data", action="store_true", dest="tensorflow_data",
                  default=False,
                  help="Use the tensorflow data loader‚")
parser.add_option("--lr", dest="learning_rate",
                  default=0.0001, type="float",
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
parser.add_option("--checkpoints", dest="checkpoints",
                  default=None, type="int",
                  help="Frequency of checkpoints to save loss and models [Default: %default]")
(options, args) = parser.parse_args()


if len(args) != 3:
    parser.error("Must provide file name, data directory and checkpoint directory")
else:
    file_name=args[0]
    data_dir = args[1]
    checkpoint_dir = args[2]

file_name = f"{file_name}_{options.learning_rate}_{options.optimizer}_{options.batch_size}_augment{options.augment}_dilation_mult_{options.dilation_rate_mult}_bn_momentum_{options.bn_momentum}_sgd_momentum_{options.momentum}_checkpoints_{options.checkpoints}"

print(file_name)


print(f"using correct residual connection")
print(f"Learning rate: {options.learning_rate}, seq-len: {options.seq_length}, randomized_data: {options.random_data}, test_data: {options.test_data}, tensorflow data: {options.tensorflow_data}, batch size: {options.batch_size}, num_epochs: {options.num_epochs}, file_name: {file_name}, shuffle: {options.shuffle}")

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


model = BasenjiModel( 
                 n_conv_layers=num_conv,
                 n_dilated_conv_layers=num_dilated_conv, 
                 conv_target_channels=conv_target_channels,
                 bn_momentum=options.bn_momentum,
                 dilation_rate_init=dilation_rate_init, 
                 dilation_rate_mult=options.dilation_rate_mult, 
                 human_tracks=options.experiments_human, 
                 mouse_tracks=options.experiments_mouse)#.to(device)


if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    
    model.to(device)

print(model)


if options.test_data: 
    print("Using test data")
    def test_dataloader(seq_length, human_tracks, mouse_tracks, batch_size, shuffle):
        train_data = {}
        for size, name in zip([200, 100], ["train", "valid"]):
            hts, mts = create_random_data(data_size=size, experiments_human=human_tracks, experiments_mouse=mouse_tracks, seq_len=seq_length)
            train_data[f"{name}_human"] = DataLoader(hts, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            train_data[f"{name}_mouse"] = DataLoader(mts, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return train_data

    dataloader = test_dataloader(options.seq_length,
                                 options.experiments_human,
                                 options.experiments_mouse, 
                                 options.batch_size, 
                                 options.shuffle)
    
elif options.tensorflow_data == True:
    from mnt.katformer.Basenji2_torch.tensorflow_dataloader import *

    # trainings data
    subset = "train"
    human_ds = BasenjiDataSet("human", subset, options.seq_length, human_fasta_path)
    mouse_ds = BasenjiDataSet("mouse", subset, options.seq_length, mouse_fasta_path)
    total = len(human_ds.region_df) # number of records
    h_train = torch.utils.data.DataLoader(human_ds, num_workers=0, batch_size=options.batch_size)
    m_train = torch.utils.data.DataLoader(mouse_ds, num_workers=0, batch_size=options.batch_size)

    # test data
    subset = "valid"
    human_test = BasenjiDataSet("human", subset, options.seq_length, human_fasta_path)
    mouse_test = BasenjiDataSet("mouse", subset, options.seq_length, mouse_fasta_path)
    h_test = torch.utils.data.DataLoader(human_test, num_workers=0, batch_size=options.batch_size)
    m_test = torch.utils.data.DataLoader(mouse_test, num_workers=0, batch_size=options.batch_size)

elif options.random_data == True:
    print("Using random data")
    dataloader = create_torch_dataloader_cross_species(options.seq_length, data_dir,
                                                        os.path.join(data_dir, "hg38.ml.fa"), os.path.join(data_dir, "mm10.ml.fa"), 
                                                        options.batch_size, shuffle=options.shuffle, random=True)

else:
    print("Using real data")
    dataloader = create_torch_dataloader_cross_species(options.seq_length, data_dir,
                                                        os.path.join(data_dir, "hg38.ml.fa"), os.path.join(data_dir, "mm10.ml.fa"), 
                                                        options.batch_size, shuffle=options.shuffle)


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
train_loss = {"human":[], "mouse":[]}
test_loss = {"human":[], "mouse":[]}
# initialize list to save correlation values
corr_val_list = []
best_loss = 1e10
best_test_loss =1e10
#corr_list, corr_list1 = [], []
#corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=5313)
for epoch in range(options.num_epochs):
    model.train()
    print(f"Starting Epoch {epoch}")

    train_loss_epoch_human, train_loss_epoch_mouse = [], []

    for i, batch in enumerate(zip(dataloader["train_human"], dataloader["train_mouse"])):
        for h, head in enumerate(["human", "mouse"]):
            model.train()
            optimizer.zero_grad()    
            global_step +=1

            # load data
            if options.augment == True:
                sequence, reverse_bool = stochastic_reverse_complement(batch[h][0], training=True)
                sequence = stochastic_shift.call(sequence, training=True)
                target = switch_reverse.call(([batch[h][1], reverse_bool]))
                sequence = sequence.to(device)
                target = target.to(device)

            else:
                sequence = batch[h][0].to(device)
                target = batch[h][1].to(device)

            # forward pass
            out = model(sequence, head)
            loss = poisson_loss(out, target)
            #print(f"loss:{loss.item()}")
            loss.mean().backward()

            # save loss
            if head == "human":
                train_loss_epoch_human.append(loss.mean().item())
            else:
                train_loss_epoch_mouse.append(loss.mean().item())
            optimizer.step()


    #if global_step % 20 == 0:

    # append the train loss for this epoch
    for loss, species in zip([train_loss_epoch_human, train_loss_epoch_mouse], ["human", "mouse"]):
        train_loss[species].append(sum(loss) / len(loss))

    #best_loss = min(train_loss["human"])
    print(f"Best loss: {best_loss}")
    print(f"New loss: {train_loss['human'][-1]}")
    if train_loss["human"][-1] <= best_loss:
        best_loss = train_loss["human"][-1]
        print("Saving best training model checkpoint")
        save_model = os.path.join(checkpoint_dir, f"{file_name}_model_training_checkpoint.pt")
        torch.save(model.state_dict(), save_model)

    # TESTING
    model.eval()
    test_loss_epoch_human, test_loss_epoch_mouse = [], []
    with torch.no_grad():
        for i, batch in enumerate(zip(dataloader["valid_human"], dataloader["valid_mouse"])):
            for h, head in enumerate(["human", "mouse"]):
                # load data
                if options.augment == True:
                    sequence, reverse_bool = stochastic_reverse_complement(batch[h][0], training=True)
                    sequence = stochastic_shift.call(sequence, training=True)
                    target = switch_reverse.call(([batch[h][1], reverse_bool]))
                    sequence = sequence.to(device)
                    target = target.to(device)

                else:
                    sequence = batch[h][0].to(device)
                    target = batch[h][1].to(device)

                out = model(sequence, head)
                loss = poisson_loss(out, target)
                if head == "human":
                    test_loss_epoch_human.append(loss.mean().item())
                else:
                    test_loss_epoch_mouse.append(loss.mean().item())
                    

        print("Computing Correlation")            
        corr_val = compute_correlation_basenji(model, device, data_dir, options.seq_length, 
                                    human_fasta_path, mouse_fasta_path, batch_size=options.batch_size,
                                    species="human", subset="valid", shuffle=True, max_steps=400)
        #corr_val = compute_correlation(model, device, human_fasta_path, organism="human", subset="test", max_steps=100)
        corr_val_list.append(corr_val)
        print("Saving the correlations")
        with open(os.path.join(checkpoint_dir, f"{file_name}_correlations.pkl"), "wb") as f:
            pickle.dump(corr_val_list, f)                    

                    

    print(f"Testing for epoch {epoch} finished")

    for loss, species in zip([test_loss_epoch_human, test_loss_epoch_mouse], ["human", "mouse"]):
        test_loss[species].append(sum(loss) / len(loss))

    print(f"Best test loss: {best_test_loss}")
    print(f"New test loss: {test_loss['human'][-1]}")
    if test_loss["human"][-1] <= best_test_loss:
        best_test_loss = test_loss["human"][-1]
        print("Saving best validation model checkpoint")
        save_model = os.path.join(checkpoint_dir, f"{file_name}_model_validation_checkpoint.pt")
        torch.save(model.state_dict(), save_model)

    # save the loss files
    print("Saving loss files")
    for file, loss in zip(["train_loss", "test_loss"],[train_loss, test_loss]):
        with open(os.path.join(checkpoint_dir,f"{file_name}_{file}.pkl"), "wb") as f:
            pickle.dump(loss, f)



