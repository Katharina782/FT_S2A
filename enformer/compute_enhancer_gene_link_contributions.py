import os
import pandas as pd
import numpy as np
import kipoiseq
#import pybedtools
import matplotlib.pyplot as plt
import seaborn as sns
from architecture_nolinear import *
from data_utils import *
import pickle


#### Run this script on the GPU cluster! It does not work on the CPU cluster!

odcf = True
if odcf: 
    data_dir = "/omics/groups/OE0540/internal/users/mikulik/master_thesis/data/gcs_basenj/"
else:
    data_dir = "/data/mikulik/mnt/gcs_basenj"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SEEN_SEQUENCE_LENGTH = 1536*128
PRED_SEQUENCE_LENGTH = 896*128

experiment_specific = False
file_name = "cell_type_agnostic"

# specify where the model is saved
model_path = os.path.join(data_dir, "models")
model_name = "enf_train_v2_batch_6_training_state_dict.pt"

# read in the set of links 
overlap = pd.read_csv(os.path.join(data_dir, "gasperini_overlap_tss_enhancers_hg38.bed"), sep="\t", header=None)
overlap.columns=["chr_gene", "start_gene", "stop_gene", "gene_name", "strand", "chr_enhancer", "start_enhancer", "stop_enhancer", "enhancer", "unknown", "overlap"]
overlap.loc[: , "enhancer_center"] = (overlap["start_enhancer"] + overlap["stop_enhancer"]) // 2
overlap["pairs"] = overlap.gene_name + "_" + overlap.enhancer
overlap.drop_duplicates("pairs", inplace=True)
assert len(overlap) == overlap.pairs.nunique()


def position_to_bin(pos, 
                    pos_type="relative",
                    target_interval=None):
    if pos_type == "absolute":
        pos = pos - target_interval.start
    elif pos_type == "relative_padded":
        pos = pos - PADDING
    return pos//128

from architecture_nolinear import * 
class EnformerContribution:
    def __init__(self, model_path, model_name, device):
        self.model = Enformer.from_hparams(
        dim = 1536,
        depth = 5,
        heads = 8,
        use_checkpointing=True,
        output_heads = dict(human = 5313, mouse= 1643),
        target_length = 896,
        )
        self.device = device
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(os.path.join(model_path, model_name), map_location=self.device))
        self.device = device
    
    def predict_on_batch(self, inputs, species):
        predictions = self.model(inputs.to(self.device))[species]
        return predictions.detach().cpu().numpy()#{k: v.numpy() for k, v in predictions.items()}
        
    def contribution_input_grad(self, input_sequence, species, target_mask):
        target_mask_mass = torch.sum(target_mask)
        input_sequence.requires_grad = True
        output = self.model(input_sequence.to(self.device))[species]
        prediction = (target_mask.to(device) * output / target_mask_mass.to(device)).sum()
        prediction.backward()
        input_sequence.requires_grad = False
        return (input_sequence.squeeze() * input_sequence.grad.squeeze()).sum(-1)
    
def position_to_bin(pos, 
                    pos_type="relative",
                    target_interval=None):
    if pos_type == "absolute":
        pos = pos - target_interval.start
    elif pos_type == "relative_padded":
        pos = pos - PADDING
    return pos//128



def get_enhancer_window(chrom, enhancer_center, gene_interval, size=2000):
    enhancer_interval = kipoiseq.Interval(chrom, enhancer_center, enhancer_center).resize(size)
    print(f"start:{enhancer_interval.start}, end: {enhancer_interval.end}")
    start_coord = np.abs(gene_interval.start - enhancer_interval.start)
    end_coord = np.abs(gene_interval.start - enhancer_interval.end)
    return start_coord, end_coord


def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()

def get_enhancer_gene_scores(contr_dict, plot = False):
    #plot = True
    link_dict = {"gene_id": [], "enhancer_id": [], "link_score": []}
    for i in range(len(contr_dict["contr"])):
        grads = contr_dict["contr"][i]
        gene_name = contr_dict["gene_stable_id"][i]
        for enhancer_id, enh_mask in contr_dict["enhancers"][i].items():
            link_score = np.abs(grads * enh_mask).sum()
            #print(link_score)
            link_dict["gene_id"].append(gene_name)
            link_dict["enhancer_id"].append(enhancer_id)
            link_dict["link_score"].append(link_score.numpy())



        if plot: 
            predictions = contr_dict["prediction"][i]
            target_interval = contr_dict["interval"][i]
            avg_pool = torch.nn.AvgPool1d(kernel_size=128, stride=128)
            pooled_contr = avg_pool(np.abs(grads).unsqueeze(0)).numpy()
            tracks = {'CAGE predictions': predictions[:, 4828],#.detach().numpy(),
                    'Enformer gradient*input': pooled_contr.squeeze()}#.detach().numpy()}
            plot_tracks(tracks, target_interval)
            plt.title(f"{gene_name}")
    return pd.DataFrame(link_dict)



# initialize the model and contribution class
contr = EnformerContribution(model_path, model_name, device)

# initialize fasta extractor
human_fasta_path = os.path.join(data_dir, "hg38.ml.fa")
fasta_extractor = FastaStringExtractor(human_fasta_path)

# initialize target mask
target_mask = np.zeros((896, 5313))
if experiment_specific:
    # get the experiment indices for K562 CAGE
    target_df = pd.read_csv(os.path.join(data_dir, "human", "targets.txt"), sep="\t", index_col=0)
    k562_index = target_df[(target_df.description.str.contains("CAGE")) & target_df.description.str.contains("K562")].index

    # add ones at the three TSS bins for K562 CAGE
    for seq_idx in [447, 448, 449]:
        for exp_idx in k562_index: # get only the CAGE tracks for K562
            target_mask[seq_idx, exp_idx] = 1

else: 
    for seq_idx in [447, 448, 449]:
        target_mask[seq_idx, :] = 1

counts = 0

contr_dict = {"gene_stable_id": [], "interval": [], "contr": [], "prediction": [], "enhancers": []}
for gene in overlap.gene_name.unique():
    print(gene)
    # get enhancer regions 
    enhancer_df = overlap[overlap.gene_name == gene]
    row = enhancer_df.iloc[0]
    if len(enhancer_df) != 0:
      contr_dict["gene_stable_id"].append(gene)

      # get 200 kbp sequence centered around TSS
      interval = kipoiseq.Interval(row.chr_gene, row.start_gene, row.stop_gene).resize(196_608)
      contr_dict["interval"].append(interval)

      # one hot encode sequence
      sequence_one_hot = torch.from_numpy(one_hot_encode(fasta_extractor.extract(interval)))

      # make a forward pass and save the prediction
      prediction = contr.predict_on_batch(sequence_one_hot, "human")
      contr_dict["prediction"].append(prediction)
      
      # make backward pass and compute input x gradient
      input_gradient = contr.contribution_input_grad(sequence_one_hot, "human", torch.from_numpy(target_mask))
      contr_dict["contr"].append(input_gradient)

      # add enhancer id and mask for 2 kbp window around enhancer center for gene i
      enh_dict = {}
      for j, row_enh in enhancer_df.iterrows():
        enhancer_mask = np.zeros(len(interval))
        idx1, idx2 = get_enhancer_window(chrom=row_enh.chr_gene, enhancer_center=row_enh.enhancer_center, gene_interval=interval)
        enhancer_mask[idx1:idx2] = 1
        print(enhancer_mask.shape)
        #assert enhancer_mask.sum() == 2000
        enh_dict[row_enh.enhancer] = enhancer_mask
        counts += 1
      contr_dict["enhancers"].append(enh_dict)

print(f"Counts: {counts}")      


link_dict = get_enhancer_gene_scores(contr_dict, plot = False)


with open(os.path.join(data_dir, f"{file_name}_enhancer_gene_link_contributions.pkl"), "wb") as f:
    pickle.dump(link_dict, f)

with open(os.path.join(data_dir, f"{file_name}_enhancer_gene_link_contributions_dict.pkl"), "wb") as f:
    pickle.dump(contr_dict, f)
