import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from data_utils_finetuning import *
import kipoiseq
import torch
from sklearn.linear_model import Ridge


def create_target_vector(count_mat, subset, overlap):
    g = {}
    log_expr = np.log(count_mat + 1)
    if ("train" in subset) & ("valid" in subset):
        print("use train and validation set")
        overlap_sub = overlap[(overlap.region_subset == subset[0]) | (overlap.region_subset == subset[1])]
    else:
        print("use test set")
        overlap_sub = overlap[overlap.region_subset == subset[0]]
    overlap_sub = overlap_sub.drop_duplicates(subset=["gene_id", "region_subset"])#, inplace=True)
    for i, row in overlap_sub.iterrows():
        g[row.gene_id] = log_expr[:, row.gene_index] # gene expression for gene "gene_id" across all cell types
    g = pd.DataFrame(g)
    print(g.shape)
    return g


def normalize_gex_predictions(gex_train, gex_test):
    # log1p normalization
    gex_train, gex_test = np.log(np.array(gex_train) + 1), np.log(np.array(gex_test) + 1)

    # subset to keep only CAGE experiments from mouse
    mouse_targets = pd.read_csv("/data/mikulik/mnt/gcs_basenj/mouse/targets.txt", header=0, sep="\t")
    cage_index = mouse_targets.description.str.contains("CAGE")
    gex_train, gex_test = gex_train[cage_index, :], gex_test[cage_index, :]

    # standardization with train mean and std
    mean, std = gex_train.mean(axis=1), gex_train.std(axis=1)
    gex_train =(gex_train - mean[:, None]) / std[:, None]
    assert np.isclose(gex_train.mean(axis=1), 0).all()
    assert np.isclose(gex_train.std(axis=1), 1).all()

    # use mean and standard deviation from training set to standardize test set, prevent data leakage!
    gex_test = (gex_test - mean[:, None]) / std[:, None]
    return gex_train, gex_test





def get_gene_expression_predictions(model,model_name, subset, seq_length, center_bin, recompute, overlap, file_name, device="cpu", mouse_fasta_path = "/data/mikulik/mnt/gcs_basenj/mm10.ml.fa"):
    if recompute:
        gex = {}
        fasta_reader = FastaStringExtractor(mouse_fasta_path)
        if len(subset) > 1:
            print(f"use {subset} set")
            overlap_sub = overlap[(overlap.region_subset == subset[0]) | (overlap.region_subset == subset[1])]
        else:
            print(f"use {subset} set")
            overlap_sub = overlap[overlap.region_subset == subset[0]]   
        overlap_sub = overlap_sub.drop_duplicates(subset=["gene_id", "region_subset"])
        #if recompute:
        for i, row in overlap_sub.iterrows():
            # create a sequence centered at the TSS of the gene
            seq = kipoiseq.Interval(row.gene_chr, row.tss, row.tss).resize(seq_length)
            seq = torch.Tensor(np.expand_dims(one_hot_encode(fasta_reader.extract(seq)), axis=0)).to(device)
            if model_name == "basenji":
                pred = model(seq, "mouse") # make prediciton with model
            if model_name == "enformer":
                pred = model(seq)["mouse"]
            # get the predicted counts at the 3 center bins
            counts = pred[:, center_bin-1:center_bin+2, :].detach().cpu().numpy().sum(axis=1).squeeze() # sum over the 3 center bins
            gex[row.gene_id] = counts
        a = pd.DataFrame(gex)
        a.to_csv(file_name, index=False, header=True)
    else:
        gex = pd.read_csv(file_name, header=0)
    return gex




def fit_ridge_regression_per_celltype(X_train, X_test, y_train, y_test, alpha=1.0):
    coeffs = np.zeros((y_test.shape[0], X_test.shape[1])) # cell types * CAGE tracks
    ridge_pred_matrix = np.zeros(y_test.shape)
    celltypes = y_test.shape[0]
    print(f"Number of celltypes: {celltypes}")
    for i in range(y_test.shape[0]):
        lr = Ridge(alpha=alpha)
        lr.fit(X_train, y_train[i, :])
        pred_test = lr.predict(X_test)
        ridge_pred_matrix [i, :] = pred_test # save predicted gene expression for test set for each celltype
        coeffs[i, :] = lr.coef_ # save coefficients for each celltype
        #print(f"ENCODE: {cage_targets.iloc[np.argmax(lr.coef_), :].description}")
        #print(f"Gastrulation celltype: {rna_target_df.iloc[i, :].celltype}")
    return coeffs, ridge_pred_matrix




def target_vector_regression(count_matrix, subset, overlap, log=True):
    target = {}
    if log:
        log_expr = np.log(count_matrix + 1)
    if subset == "test":
        overlap_sub = overlap[overlap.region_subset == subset]
    else:
        overlap_sub = overlap[(overlap.region_subset == subset[0]) | (overlap.region_subset == subset[1])]
        overlap_sub.drop_duplicates(subset="gene_id", keep="first", inplace=True) # remove any duplicated genes
    for i, row in overlap_sub.iterrows():
        target[row.gene_id] = log_expr[:, row.gene_index] # gene expression for gene "gene_id" across all cell types
    target = pd.DataFrame(target)
    print(f"The training set contains {target.shape[1]} genes.")
    return target


def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    #plt.axvline(x=bin,color="red")
    plt.show()


def get_cage_predictions(data_dir, file_name, overlap, subset, seq_length=131_072, center_bin=896//2, save=True, mouse_fasta_path = "/data/mikulik/mnt/gcs_basenj/mm10.ml.fa"):
    fasta_reader = FastaStringExtractor(mouse_fasta_path)
    gex_basenji = {}

    if subset == "test":
        overlap_sub = overlap[overlap.region_subset == subset]
    else:
        overlap_sub = overlap[(overlap.region_subset == subset[0]) | (overlap.region_subset == subset[1])]

    #iterate over all genes   
    for i, row in overlap_sub.iterrows():
        # create a sequence centered at the TSS of the gene
        seq = kipoiseq.Interval(row.gene_chr, row.tss, row.tss).resize(seq_length)
        seq = torch.Tensor(np.expand_dims(one_hot_encode(fasta_reader.extract(seq)), axis=0))
        pred = basenji_model(seq, "mouse") # make prediciton with basenji
        # get the predicted counts at the 3 center bins
        counts = pred[:, center_bin-1:center_bin+2, :].detach().numpy().sum(axis=1).squeeze()
        gex_basenji[row.gene_id] = counts
    a = pd.DataFrame(gex_basenji)

    a.to_csv(os.path.join(data_dir, file_name), index=False, header=True)
    return gex_basenji