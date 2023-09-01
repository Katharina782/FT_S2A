from torchmetrics import Metric
from typing import Optional
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from data_utils import *
from data_utils_finetuning import * 


## Plot Functions

# loss cruves of training Basenji2
def plot_loss_curves(data_dir, file_name, species = ["human", "mouse"]):
    with open(f"{data_dir}{file_name}_test_loss.pkl", "rb") as f:
        test_loss = pickle.load(f)
    with open(f"{data_dir}{file_name}_train_loss.pkl", "rb") as f:
        train_loss = pickle.load(f)
    for i in species:
        plt.plot(np.arange(len(train_loss[f"{i}"])), train_loss[f"{i}"], label=f"{i} train")
        plt.plot(np.arange(len(test_loss[f"{i}"])), test_loss[f"{i}"], label=f"{i} valid")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.show()

# loss curves of fine-tuning experiments with Basenji2
def plot_loss_curves_finetune(data_dir, file_name, species = ["mouse"], ylim=None, xlim=None, color1="darkblue", color2="darkgreen", save_pdf=None):
    with open(f"{data_dir}{file_name}_test_loss.pkl", "rb") as f:
        test_loss = pickle.load(f)
    with open(f"{data_dir}{file_name}_train_loss.pkl", "rb") as f:
        train_loss = pickle.load(f)
    for i in species:
        plt.plot(np.arange(len(train_loss[f"{i}"])), train_loss[f"{i}"], label=f"{i} train", color=color1)
        plt.plot(np.arange(len(test_loss[f"{i}"])), test_loss[f"{i}"], label=f"{i} valid", color=color2)
        plt.legend()
        plt.title(f"{file_name}")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        if ylim!=None:
            plt.ylim(ylim)
        if xlim!=None:
            plt.xlim(xlim)
        if save_pdf != None:
            plt.savefig(os.path.join(save_pdf,  f'{file_name}_loss.pdf'), format='pdf')
        plt.show()

# separate loss curves for rna and atac
def plot_loss_curves_separate(data_dir, file_name, species = ["human", "mouse"]):
    with open(f"{data_dir}{file_name}_test_loss_separate.pkl", "rb") as f:
        test_loss = pickle.load(f)
    with open(f"{data_dir}{file_name}_train_loss_separate.pkl", "rb") as f:
        train_loss = pickle.load(f)
    #print(len(train_loss["human"]), len(test_loss["human"]))
    #print(train_loss_step)
    #print(train_loss_step)
    for loss in [train_loss, test_loss]:
        plt.plot(np.arange(len(loss["rna"])), loss["rna"], label=f"rna")
        plt.plot(np.arange(len(loss["atac"])), loss["atac"], label=f"atac")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel(f"Loss {loss}")
        plt.show()

# correlation curves during Basenji2 training
def plot_correlation_curves(data_dir, file_name, ylim=None, xlim=None):
    with open(f"{data_dir}{file_name}_correlations.pkl", "rb") as f:
        corr = pickle.load(f)
    max_corr = np.array(corr).max()
    plt.plot(np.arange(len(corr)), corr)
    plt.xlabel("Steps")
    plt.ylabel("Pearson Correlation")
    plt.title(f"Maximum correlation: {max_corr}")
    if ylim!=None:
        plt.ylim(ylim)
    if xlim!=None:
        plt.xlim(xlim)
    plt.show()


# stratify correaltion across positions by the expeirmental type (CAGE/TF-CHIP/Histone-CHIP/DNase)
def plot_corr_per_experiment(corr_tensor, data_dir="/data/mikulik/mnt/gcs_basenj/", species="human"):
    target_df = pd.read_csv(os.path.join(data_dir, species, "targets.txt"), sep="\t", header=0)
    target_df[["experiment", "descr"]] = target_df.description.str.split(":", n=1, expand=True)
    for i in target_df.experiment.unique():
        if i == "CHIP":
            sub_histone = target_df[target_df.description.str.contains("H3K")].index
            sub_tf = target_df[(~target_df.description.str.contains("H3K")) & (target_df.experiment == "CHIP")].index
            for ind, name in zip([sub_histone, sub_tf], ["Histone-CHIP", "TF-CHIP"]):
                sub = corr_tensor[ind]
                sns.histplot(sub)
                mean_corr = np.round(sub.mean(), decimals=2)
                print(mean_corr)
                plt.title(f"{name}, mean:{mean_corr:.2f}")
                plt.show()
        else: 
            index = target_df[target_df["experiment"] == i].index
            sub = corr_tensor[index]
            sns.histplot(sub)
            mean_corr = np.round(sub.mean(), decimals=2)
            print(mean_corr)
            plt.title(f"{i}, mean:{mean_corr:.2f}")
            plt.show()

# histogram of correlation values at the gene TSS
def plot_tss_corr_hist(corr_vec, exp="CAGE", model=None, title=None, color="green"):
    sns.histplot(corr_vec, color=color)
    mean_corr = np.round(corr_vec.mean(), decimals=2)
    plt.title(f"{title}, {model},{exp}")
    plt.xlabel(f"Correlation,  mean:{mean_corr:.3f}")
    plt.show()



# function from deepmind
def plot_tracks(tracks, interval, height=1.5, color="darkred"):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y, color=color)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()


# plot correlation values of differen parameter settings for fine-tuning Basenji2 model
def plot_parameter_comparison(lr_dict, title=None, save_pdf=None, color_palette="Dark2"):
    lr_df = pd.DataFrame.from_dict(lr_dict, orient='index', columns=['Value'])
    lr_df.reset_index(inplace=True)
    lr_df.columns=['LR', 'corr']
    lr_df.sort_values("corr", inplace=True)
    sns.barplot(data=lr_df, x="LR", y="corr", palette=sns.color_palette("tab20", 20))#sns.color_palette("Paired"))
    #sns.set_palette = sns.color_palette("Paired")
    #sns.barplot(data=lr_df, x="LR", y="corr")#, color=)
    plt.xticks(rotation=90)
    if title != None:
        plt.title(title)
    if save_pdf != None:
        plt.savefig(save_pdf, format='pdf')
    plt.show()

# correlation values for each cell types vs number of single-cell representatives of that cell type
# used for evaluating fine-tuned Basenji2 models
def corr_vs_n_cells(target_df, model, title=None, ylabel=None, save_pdf=None):
    unique_colors = target_df['color'].unique()
    # Create a custom color palette
    custom_palette = sns.color_palette(unique_colors)
    # Set the custom palette
    sns.set_palette = custom_palette
    # create scatterplot
    sns.scatterplot(target_df, x="n", y=model, hue="celltype", s=250, palette=custom_palette)
    plt.title(f"{title}, {model}, µ = {target_df[model].mean():.3f}")
    plt.xlabel("Number of cells for a cell type")
    if ylabel != None:
        plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.legend([])
    if save_pdf != None:
        plt.savefig(save_pdf, format='pdf')
    plt.show()


# Correlation computation along positions from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/metrics.py
from torchmetrics import Metric
class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable: Optional[bool] = False
    full_state_update:bool = False
    higher_is_better: Optional[bool] = True
    def __init__(self, n_channels:int, dist_sync_on_step=False):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        # we get a vector of length num_channels * region_positions -> correlation
        # these vectors are aggregated (summed) over several input sequences depending on the number of iteartions/valid sequences
        super().__init__(dist_sync_on_step=dist_sync_on_step, full_state_update=False)
        self.reduce_dims=(0, 1)
        # name of variable, default value of state = tensor (will be reset to this value when self.reset() is called.
        # dist_reduce_fx = function to reduce state -> use torch.sum/torch.mean/etc.
        self.add_state("product", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("count", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        # this aggregates the product of target * prediction across channels and region positions
        self.product += torch.sum(preds * target, dim=self.reduce_dims) 
        # sum across channels and region positions
        self.true += torch.sum(target, dim=self.reduce_dims)
        # sum of squared values across channels and region positions
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims) # num_channels * seq_len (5313 * 896)

    def compute(self):
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = (self.product
                    - true_mean * self.pred
                    - pred_mean * self.true
                    + self.count * true_mean * pred_mean)

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation


# compute pearson correlation between columns of two matrices and return a vector of length number of columns
def pearson_corr(x, y, per_column = True, log = True):
    if log:
        x, y = np.log(x + 1), np.log(y + 1)
    assert x.shape == y.shape
    if per_column:
        rows, cols = 0, 1
    else:
        rows, cols = 1, 0 
    corr_vec = np.zeros(x.shape[cols]) 
    row, col = x.shape[rows], x.shape[cols]
    for column in range(col):
        a, b = x[:, column], y[:, column]
        sum = np.sum((a- np.mean(a)) * (b - np.mean(b)))
        corr = (1/row * sum) / (np.sqrt(np.var(a)) * np.sqrt(np.var(b)))
        corr_vec[column] = corr
    return corr_vec



def standardize_matrix_columns(mat, log=False):
    '''
    Standardize each column of the matrix to have mean 0 and std 1. 
    Z-score for each experiment across genes to normalize counts at the TSS. 
    
    mat(numpy array): matrix of dimension genes x cells or genes x experiments
    log(bool): if True, use log1p transformation before standardization
    '''
    if log == True:
        mat = np.log(mat + 1)
    norm_mat = (mat - np.mean(mat, axis=0)) / np.std(mat, axis=0)
    # make sure that mean is zero now
    assert np.all(np.isclose(norm_mat.mean(axis=0), 0))
    # make sure that std is one now
    assert np.all(np.std(norm_mat, axis=0)) == 1
    return norm_mat


def rna_atac_correlation(data_dir, model, device="cpu",subset="valid", max_steps=400, atac_channels=35, rna_channels=37):
    '''
    Compute the correlation across positions for RNA-seq and ATAC-seq target tracks separately.

    data_dir(str): path to data directory
    model(torch.nn.Module): trained/finetuned model to evaluate
    device(torch.device): one of "cpu" or "cuda", device used for training
    subset(str): one of train/valid/test
    max_steps(int): number of input sequences to use for evaluation
    atac_channels(int): number of ATAC-seq output tracks
    rna_channels(int):  number of RNA-seq output tracks
    '''
    corr_coef_atac = MeanPearsonCorrCoefPerChannel(n_channels=atac_channels)
    corr_coef_rna = MeanPearsonCorrCoefPerChannel(n_channels=rna_channels)
    dl = create_dataloader_gastr_norm(seq_length=131072,
                                    atac=True, 
                                    rna=True, 
                                    data_dir=data_dir,
                                    human_fasta_path=os.path.join(data_dir, "hg38.ml.fa"), 
                                    mouse_fasta_path=os.path.join(data_dir, "mm10.ml.fa"), 
                                    batch_size=1, 
                                    rna_data="basenji2_gastrulation_dataset_rna_grcm3_final",  
                                    log_transform_rna=False, 
                                    shuffle=False, 
                                    random=False, 
                                    distributed_sampler=False)
    # for each seqeunce of the subset make a prediction and compare to observation
    for i, batch in enumerate(tqdm(dl[subset])):
        if max_steps > 0 and i >=max_steps:
            break
        sequence = batch[0].to(device)
        target = batch[1].to(device)
        target_atac, target_rna = target[:, :, :atac_channels], target[:, :, -rna_channels:]
        with torch.no_grad():
            pred = model(sequence)
            pred_atac, pred_rna = pred[:, :, :atac_channels], pred[:, :, -rna_channels:]
            corr_coef_atac(preds=pred_atac.cpu(), target=target_atac.cpu())
            corr_coef_rna(preds=pred_rna.cpu(), target=target_rna.cpu())
    atac_corr_vec = corr_coef_atac.compute()
    rna_corr_vec = corr_coef_rna.compute()
    return atac_corr_vec, rna_corr_vec


# get correlation at gene TSS across genes or across experiments 
def get_corr_from_gene_dict(target_df, data_dir=None, file_name=None, gene_dict=None, per_exp=True, log=True, standardize_exp=True):
    '''
    This function takes a dictionary with genes as keys and vectors of length #tracks as values.
    Each vector contains the predicted/observed counts at the TSS for each gene respectively.

    target_df(pd.DataFrame): contains a description of the output target tracks
    data_dir(str): path to data directory of saved gene dictionary
    file_name(str): name of the file containing the gene dictionary, this file will be loaded
    gene_dict(dict): if data_dir and file_name are both None, the function uses a loaded dictionary as input    
    per_exp(bool): if True, compute correlation per experiment across genes, else compute correlation per gene across experiments
    log(bool): if True, log-transform the counts before computing the correlation
    standardize_exp(bool): if True, standardize the counts for each experiment across genes before computing the correlation
    '''

    # read from file
    if (data_dir != None) & (file_name!=None):
        with open(os.path.join(data_dir, file_name), "rb") as f:
            gene_dict = pickle.load(f)
        pred, tar = np.array(list(gene_dict["pred"].values())), np.array(list(gene_dict["tar"].values()))
    # read from dict
    elif gene_dict != None:
        pred, tar = np.array(list(gene_dict["pred"].values())), np.array(list(gene_dict["tar"].values()))

    # subset for only CAGE tracks
    index = target_df.description.str.contains("CAGE")
    tar, pred = tar[:, index], pred[:, index]
    print(tar.shape, pred.shape)
    
    if per_exp: 
        print(f"Computing correlation per experiment across genes, log_transforming: {log}")
        corr_vec = pearson_corr(pred, tar, per_column=True, log=log)
    else:
        print(f"Computing correlation per gene across experiments, standardizeing across genes for each experiment first, log-transforming: {log}")
        # standardize for each experiment across genes & log-transform
        if standardize_exp:
            pred, tar = standardize_matrix_columns(pred, log=log), standardize_matrix_columns(tar, log=log)
        corr_vec = pearson_corr(pred.T, tar.T, per_column=True, log=False)
    return corr_vec


def update_gene_dict(single_region_df:pd.DataFrame, gene_dict:dict, target:np.array, prediction:np.array):
    '''
    This function updates the gene dictionary with the predicted/observed counts at the TSS for each gene respectively.

    single_region_df(pd.DataFrame): contains all TSS overlapping a certain input region
    gene_dict(dict): dictionary with genes as keys and vectors of length #tracks as values
    target(np.array): array of shape 1 x 896 x #tracks that contains observed RNA-seq counts
    prediction(np.array): array of shape 1 x 896 x #tracks that contains predicted RNA-seq counts
    '''
    for i, row in single_region_df.iterrows():
        gene = row.gene_id
        bin = int(row.bin)
        counts_target = target[:, bin-1 : bin+2, :].sum(axis=1).squeeze()
        counts_prediction = prediction[:, bin-1 : bin+2, :].sum(axis=1).squeeze()
        gene_dict["pred"][gene] = gene_dict["pred"][gene] + counts_prediction
        gene_dict["tar"][gene] = gene_dict["tar"][gene] + counts_target


        
# When computing correlation of gene expression at TSS
def get_overlap(data_dir, file_name):
    '''
    Load the dataframe with all gene TSSs overlapping one of the test/valid/train input regions. The dataframe
    contains TSS coordinates, and region coordinates, region index, as well as, which subset (trian/test/valid) the region belongs to.

    data_dir(str): path to data directory where the dataframe is saved
    file_name(str): name of the file containing the dataframe
    '''
    overlap_tss = pd.read_csv(os.path.join(data_dir, f"{file_name}.bed"), sep="\t", header=None)
    overlap_tss.columns = ["tss_chrom", "tss_start", "tss_end", "gene_id",  "test_region_chr", "test_region_start", "test_region_end", "subset", "region_index",  "overlap_size"]
    overlap_tss["region_coord"] = overlap_tss.test_region_chr + ":" + overlap_tss.test_region_start.astype("str") + "_" + overlap_tss.test_region_end.astype("str")
    overlap_tss["tss_coord"] = overlap_tss.tss_chrom + ":" + overlap_tss.tss_start.astype("str") + "_" + overlap_tss.tss_end.astype("str")
    overlap_tss["seq_len"] = overlap_tss.test_region_end -  overlap_tss.test_region_start # length of the input region
    # calculate the position of the TSS within the test region
    overlap_tss["pos_tss"] = np.abs(overlap_tss.test_region_start - overlap_tss.tss_start)
    assert 0 < overlap_tss.pos_tss.max() < 114688 
    # convert the position of the TSS to the 128 bp bin number
    overlap_tss["bin"] = np.round(overlap_tss.pos_tss // 128)
    assert all(overlap_tss.bin >= 0)

    # sort values by region index 
    overlap_tss  = overlap_tss.sort_values(by="region_index", ascending=True)
    assert overlap_tss.bin.max() < 896
    assert overlap_tss.bin.min() >= 0
    return overlap_tss



def get_gene_dict_for_correlation(model, species, ds, overlap_tss, device, file_name, data_dir, save=False, tracks=None):
    '''
    For each region index in the dataset that overlaps with at least one gene TSS, collect the RNA-seq counts at the TSS for each gene.

    model(torch.nn.Module): trained/finetuned model to evaluate
    species(str): one of human/mouse
    ds(torch.utils.data.Dataset): dataset object
    overlap_tss(pd.DataFrame): dataframe with all gene TSSs coordinates overlapping one of the test/valid/train input regions
    device(torch.device): one of "cuda" or "cpu", device used for computations
    data_dir(str): path to data directory where the dataframe is saved
    file_name(str): name of the file to save the gene dictionary
    save(bool): if True, save the gene dictionary to a file
    tracks(int): Number of output tracks of the model to evaluate. If None, use default number for mouse and human, else use custom number of tracks.

    '''
    # initialize an empty dictionary
    # for each gene (key) I save a vector of lenght #tracks which contains the predictions/targets for each gene respectively
    if tracks != None:
        tracks = tracks
    else:
        tracks = (5313 if species == "human" else 1643)

    gene_dict = {"pred": {gene: np.zeros(tracks) for gene in overlap_tss.gene_id.unique()},
                "tar": {gene: np.zeros(tracks) for gene in overlap_tss.gene_id.unique()}}

    # one input sequence may contain several TSSs of several genes
    by_region = overlap_tss.groupby("region_index")

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
                pred = model(seq.to(device), species)
            tar, pred = np.expand_dims(tar,axis=0), pred.detach().cpu().numpy()

            update_gene_dict(single_region_df=by_region.get_group(index), gene_dict=gene_dict, target=tar, prediction=pred)
        else:
            continue

    print("Saving gene dictionary")

    if save:
        with open(os.path.join(data_dir, "basenji_training", "correlations", f"{file_name}_gene_dict.pkl"), "wb") as f:
            pickle.dump(gene_dict, f)
    return gene_dict