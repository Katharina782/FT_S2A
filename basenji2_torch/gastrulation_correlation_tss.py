
from data_augmentation import *
from data_utils_finetuning import *
import pandas as pd
import kipoiseq
import numpy as np

def collect_counts_tss(data_dir, seq_length, subset, model, overlap, count_mat, device="cpu", n_tracks=37, joint_training=False, from_scratch=False):
    mouse_fasta_path = os.path.join(data_dir, "mm10.ml.fa")
    fasta_reader = FastaStringExtractor(mouse_fasta_path)
    center_bin = 896//2
    overlap_sub = overlap[overlap.region_subset == subset]

    # prepare empty gene dictionary
    gene_dict = {"pred": {gene: np.zeros(n_tracks) for gene in overlap_sub.gene_id.unique()},
                "tar": {gene: np.zeros(n_tracks) for gene in overlap_sub.gene_id.unique()}}

    # load normalized count matrix for pseudobulks
    #count_mat = np.load(os.path.join(data_dir, "mouse_rna_seq_gastrulation_processing", "rna_lib_scaled.npy"))         
       
    for i, row in overlap_sub.iterrows():
        seq = kipoiseq.Interval(row.gene_chr, row.tss, row.tss).resize(seq_length)
        seq = torch.Tensor(np.expand_dims(one_hot_encode(fasta_reader.extract(seq)), axis=0)).to(device)
        with torch.no_grad():
            if from_scratch:
                pred = model(seq, "mouse")
            else:
                pred = model(seq)
        if joint_training:
            pred = pred[:, :, -37:]
        #print(pred.shape)
        counts = pred[:, center_bin-1:center_bin+2,: ].to("cpu").numpy().sum(axis=1).squeeze()
        gene_dict["pred"][row.gene_id] = counts
        gene_dict["tar"][row.gene_id] = count_mat[:, row.gene_index]
    return gene_dict


# compute correlation at TSS
def pearson_corr(x, y,  log = True):
    if log:
        x, y = np.log(x + 1), np.log(y + 1)
    assert x.shape == y.shape
    rows, cols = 0, 1
    corr_vec = np.zeros(x.shape[cols]) 
    row, col = x.shape[rows], x.shape[cols]
    for column in range(col):
        a, b = x[:, column], y[:, column]
        sum = np.sum((a- np.mean(a)) * (b - np.mean(b)))
        corr = (1/row * sum) / (np.sqrt(np.var(a)) * np.sqrt(np.var(b)))
        corr_vec[column] = corr
    return corr_vec

# To normalize across genes for each experiment
def standardize_matrix_columns(mat, log=False):
    if log == True:
        mat = np.log(mat + 1)
    norm_mat = (mat - np.mean(mat, axis=0)) / np.std(mat, axis=0)
    # make sure that mean is zero now
    assert np.all(np.isclose(norm_mat.sum(axis=0), 0))
    # make sure that std is one now
    assert np.all(np.std(norm_mat, axis=0)) == 1
    return norm_mat

# get correlation vector across genes or across experiments
def get_corr_from_gene_dict(data_dir=None, file_name=None, gene_dict=None, per_exp=True, log=True, standardize_exp=True):
    if (data_dir != None) & (file_name!=None):
        with open(os.path.join(data_dir, file_name), "rb") as f:
            gene_dict = pickle.load(f)
        pred, tar = np.array(list(gene_dict["pred"].values())), np.array(list(gene_dict["tar"].values()))
    elif gene_dict != None:
        pred, tar = np.array(list(gene_dict["pred"].values())), np.array(list(gene_dict["tar"].values()))
    if per_exp: 
        print(f"Computing correlation per experiment across genes, log_transforming: {log}")
        corr_vec = pearson_corr(pred, tar, log=log)
    else:
        print(f"Computing correlation per gene across experiments, standardizeing across genes for each experiment first, log-transforming: {log}")
        # standardize for each experiment across genes & log-transform
        if standardize_exp:
            pred, tar = standardize_matrix_columns(pred, log=log), standardize_matrix_columns(tar, log=log)
            corr_vec = pearson_corr(pred.T, tar.T, log=False)
        else:
            corr_vec = pearson_corr(pred.T, tar.T, log=log)

    return corr_vec