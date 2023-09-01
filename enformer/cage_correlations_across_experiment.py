# %%
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
from data_augmentation import *
from data_utils import *


import pickle
import os
#import tensorflow as tf






# %%

def pearson_corr(x, y, log = True):
    if log:
        x, y = np.log(x + 1), np.log(y + 1)
    assert x.shape == y.shape
    rows, cols = 0, 1
    corr_vec = np.zeros(x.shape[cols]) 
    row, col = x.shape[rows], x.shape[cols]
    for column in range(col):
        a, b = x[:, column], y[:, column]
        corr = np.corrcoef(a, b)[0,1]
        corr_vec[column] = corr
    return corr_vec

from collections import namedtuple



class CageCorr():
    def __init__(self, data_dir, train, test, valid):
        GeneSets = namedtuple("GeneSets", ["train", "test", "valid"])
        self.number_genes = {"train": 0, "test": 0, "valid": 0}
        self.gene_set = GeneSets(train, test, valid)
        self.data_dir = data_dir

    def get_gene_exp_matrix(self):
        tmp = {}
        self.gene_ids= {}#{"train": [], "test": [], "valid": []}
        for name, file in self.gene_set._asdict().items():
            with open(os.path.join(self.data_dir, file), "rb") as f:
                gene_dict = pickle.load(f)
            self.gene_ids[name] = list(gene_dict["pred"].keys())
            pred, tar = np.array(list(gene_dict["pred"].values())), np.array(list(gene_dict["tar"].values()))
            self.number_genes[name] += pred.shape[0]
            tmp[file] = (pred, tar)
        pred_list, tar_list = [item[0] for item in tmp.values()], [item[1] for item in tmp.values()]
        self.raw_pred, self.raw_tar = np.concatenate(pred_list, axis=0), np.concatenate(tar_list, axis=0)
      
    
    def normalize_gene_mat(self, target_df_dir, cage=True, z_score=True):
        self.get_gene_exp_matrix()
        target_df = pd.read_csv(target_df_dir, sep="\t")
        if cage:
            index = target_df.description.str.contains("CAGE")
            # subset to only CAGE tracks
            pred, tar = self.raw_pred[:, index].copy(), self.raw_tar[:, index].copy()
            pred, tar = np.log(pred + 1), np.log(tar + 1)
            # z-score for each CAGE experiment across all genes
            self.pred, self.tar = (pred - pred.mean(axis=0)) / pred.std(axis=0), (tar - tar.mean(axis=0)) / tar.std(axis=0)
            assert np.isclose(self.pred.mean(axis=0), 0).all()
            assert np.isclose(self.pred.std(axis=0),1).all()
            assert np.isclose(self.tar.mean(axis=0), 0).all()
            assert np.isclose(self.tar.std(axis=0),1).all()
            return self.pred, self.tar
    
        else:
            pred, tar = self.raw_pred.copy(), self.raw_tar.copy()
            if z_score:
                # log(x + 1)
                pred, tar = np.log(pred + 1), np.log(tar + 1)
                # z-score for each CAGE experiment across all genes
                self.pred, self.tar = (pred - pred.mean(axis=0)) / pred.std(axis=0), (tar - tar.mean(axis=0)) / tar.std(axis=0)
                #assert np.isclose(self.pred.mean(axis=0), 0).all()
                #assert np.isclose(self.pred.std(axis=0),1).all()
                #assert np.isclose(self.tar.mean(axis=0), 0).all()
                assert np.isclose(self.tar.std(axis=0),1).all()
                return self.pred, self.tar
            else:
                self.pred, self.tar = np.log(pred + 1), np.log(tar + 1)
                return self.pred, self.tar
    
    def get_subset(self, subset="test"):
        if subset == "test":
            return self.pred[self.number_genes["train"]: -self.number_genes["valid"], :], self.tar[self.number_genes["train"]: -self.number_genes["valid"], :]

        if subset == "train":
            return self.pred[:self.number_genes["train"], :], self.tar[:self.number_genes["train"], :]
        
        if subset == "valid":
            return self.pred[-self.number_genes["valid"]:, :], self.tar[-self.number_genes["valid"]:, :]
        
    def across_gene_correlation(self, subset=None):
        if subset != None:
            print(f"Compute correlation across genes for {subset}")
            pred, tar = self.get_subset(subset)
            return pearson_corr(pred, tar, log=False)
        else:
            print(f"Compute correlation across genes for all subsets")
            return pearson_corr(self.p, self.t, log=False)
    def across_experiment_correlation(self, subset=None):
        if subset != None:
            print(f"Compute correlation across experiments for {subset}")
            pred, tar = self.get_subset(subset)
            return pearson_corr(pred.T, tar.T, log=False)
        else:
            print(f"Compute correlation across experiments for all subsets")
            return pearson_corr(self.p.T, self.t.T, log=False)



# Deprecated
class CageCorrOld():
    def __init__(self, data_dir, train, test, valid):
        GeneSets = namedtuple("GeneSets", ["train", "test", "valid"])
        self.number_genes = {"train": 0, "test": 0, "valid": 0}
        self.gene_set = GeneSets(train, test, valid)
        self.data_dir = data_dir

    def get_gene_exp_matrix(self):
        tmp = {}
        for name, file in self.gene_set._asdict().items():
            with open(os.path.join(self.data_dir, file), "rb") as f:
                gene_dict = pickle.load(f)
            pred, tar = np.array(list(gene_dict["pred"].values())), np.array(list(gene_dict["tar"].values()))
            self.number_genes[name] += pred.shape[0]
            tmp[file] = (pred, tar)
        pred_list, tar_list = [item[0] for item in tmp.values()], [item[1] for item in tmp.values()]
        self.raw_pred, self.raw_tar = np.concatenate(pred_list, axis=0), np.concatenate(tar_list, axis=0)
      
    
    def normalize_gene_mat(self, target_df_dir, cage=True):
        self.get_gene_exp_matrix()
        target_df = pd.read_csv(target_df_dir, sep="\t")
        if cage:
            index = target_df.description.str.contains("CAGE")
            # subset to only CAGE tracks
            pred, tar = self.raw_pred[:, index], self.raw_tar[:, index]

        # log(x + 1)
        pred, tar = np.log(pred + 1), np.log(tar + 1)
        # z-score for each CAGE experiment across all genes
        self.pred, self.tar = (pred - pred.mean(axis=0)) / pred.std(axis=0), (tar - tar.mean(axis=0)) / tar.std(axis=0)
        assert np.isclose(self.pred.mean(axis=0), 0).all()
        assert np.isclose(self.pred.std(axis=0),1).all()
        assert np.isclose(self.tar.mean(axis=0), 0).all()
        assert np.isclose(self.tar.std(axis=0),1).all()
        return self.pred, self.tar
    
    def get_subset(self, subset="test"):
        if subset == "test":
            return self.pred[self.number_genes["train"]: -self.number_genes["valid"], :], self.tar[self.number_genes["train"]: -self.number_genes["valid"], :]

        if subset == "train":
            return self.pred[:self.number_genes["train"], :], self.tar[:self.number_genes["train"], :]
        
        if subset == "valid":
            return self.pred[-self.number_genes["valid"]:, :], self.tar[-self.number_genes["valid"]:, :]
        
    def across_gene_correlation(self, subset=None):
        if subset != None:
            print(f"Compute correlation across genes for {subset}")
            pred, tar = self.get_subset(subset)
            return pearson_corr(pred, tar, log=False)
        else:
            print(f"Compute correlation across genes for all subsets")
            return pearson_corr(self.p, self.t, log=False)
    def across_experiment_correlation(self, subset=None):
        if subset != None:
            print(f"Compute correlation across experiments for {subset}")
            pred, tar = self.get_subset(subset)
            return pearson_corr(pred.T, tar.T, log=False)
        else:
            print(f"Compute correlation across experiments for all subsets")
            return pearson_corr(self.p.T, self.t.T, log=False)

        

