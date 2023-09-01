
import pickle
import os
from data_utils import *



def cage_corr_tensorflow(overlap_tss, region_df, ds):
    assert np.all(overlap_tss.region_index.isin(region_df["index"]))
    by_region = overlap_tss.groupby("region_index")
    # initialize an empty gene dictionary
    gene_dict = {"pred": {gene: np.zeros(5313) for gene in overlap_tss.gene_id.unique()}, 
                "tar": {gene: np.zeros(5313) for gene in overlap_tss.gene_id.unique()}}
    ds = iter(ds)
    count = 0
    for i, index in enumerate(test_regions["index"]):
            count+=1
            print(count)
            batch = next(ds)
            if overlap_tss[overlap_tss.region_index == index].shape[0] >= 1:
                tar = tf.expand_dims(batch["target"], axis=0)
                preds = predict(tf.expand_dims(batch["sequence"], axis=0))
                update_gene_dict(single_region_df=by_region.get_group(index), gene_dict=gene_dict, target=tar.numpy(), prediction=preds.numpy())

    return gene_dict
    #yield gene_dict 

def update_gene_dict(single_region_df:pd.DataFrame, gene_dict:dict, target:np.array, prediction:np.array):
    for i, row in single_region_df.iterrows():
        gene = row.gene_id
        bin = int(row.bin)
        counts_target = target[:, bin-1 : bin+2, :].sum(axis=1).squeeze()
        counts_prediction = prediction[:, bin-1 : bin+2, :].sum(axis=1).squeeze()
        gene_dict["pred"][gene] = gene_dict["pred"][gene] + counts_prediction
        gene_dict["tar"][gene] = gene_dict["tar"][gene] + counts_target


def get_corr_from_preds(gene_dict, log10 = False):        
    genes_tss_experiments_pred = np.array(list(gene_dict["pred"].values()))
    genes_tss_experiments_tar = np.array(list(f["tar"].values()))
    if log10 == True:
        genes_tss_experiments_pred = np.log10(genes_tss_experiments_pred + 1)
        genes_tss_experiments_tar = np.log10(genes_tss_experiments_tar + 1)
    corr_vec = pearson_corr(genes_tss_experiments_pred, genes_tss_experiments_tar, per_column=True)
    return corr_vec



odcf = False
#pc006 = True
if odcf:
    data_dir = "/omics/groups/OE0540/internal/users/mikulik/master_thesis/data/gcs_basenj/"
    from data_utils import * 
    
    
else:
    data_dir = "/data/mikulik/mnt/gcs_basenj/"
    from deepmind_utils import * 

print(data_dir)


# the trained enformer model is available under:
model_path = 'https://tfhub.dev/deepmind/enformer/1'



#specify device cpu, because the model does not fit on the GPU
tf.device('/cpu:0')
SEQUENCE_LENGTH = 393216
TARGET_LENGTH = 896
BIN_SIZE = 128
experiments_human = 5313
experiments_mouse = 1643


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
tf.config.experimental.list_physical_devices('CPU')
tf.device('/CPU:0')


with tf.device("CPU"):
    enformer_model = hub.load("https://tfhub.dev/deepmind/enformer/1").model#, options=tf_obj).model



target_df = pd.read_csv(os.path.join(data_dir, "human", "targets.txt"), sep = "\t")
target_df[["experiment", "descr"]] = target_df.description.str.split(":", n=1, expand=True)



#def get_organism_path(organism):
 #   return os.path.join("/omics/groups/OE0540/internal/users/mikulik/master_thesis/data/gcs_basenj/", organism)
human_fasta_path = f"{data_dir}hg38.ml.fa"
ds = DeepMindData("human", "test", seq_len=393216, fasta_path=human_fasta_path)
def predict(x):
    return enformer_model.predict_on_batch(x)["human"]



#overlap_tss = get_overlap(data_dir, "tss_overlap_all")
#region_df = pd.read_csv(os.path.join(data_dir, "human", "sequences.bed"), sep="\t", header=None)
#region_df.reset_index(inplace=True)
#region_df.columns = ["index", "chrom", "start", "end", "subset"]
#assert np.all(overlap_tss.region_index.isin(region_df["index"]))
#by_region = overlap_tss.groupby("region_index")

protein_coding = True

if protein_coding:
    overlap_tss = get_overlap(data_dir=data_dir, file_name="tss_overlap_protein_coding")
else:
    overlap_tss = get_overlap(data_dir, "tss_overlap_all")
region_df = pd.read_csv(os.path.join(data_dir, "human", "sequences.bed"), sep="\t", header=None)
region_df.reset_index(inplace=True)
region_df.columns = ["index", "chrom", "start", "end", "subset"]
assert np.all(overlap_tss.region_index.isin(region_df["index"]))
by_region = overlap_tss.groupby("region_index")
test_regions = region_df[region_df.subset == "test"]
print(len(test_regions))
assert np.all(overlap_tss.region_index.isin(test_regions["index"]))



#gene_dict_deepmind = cage_corr_tensorflow(overlap_tss, region_df, ds)
gene_dict_deepmind = cage_corr_tensorflow(overlap_tss, test_regions, ds)



with open(os.path.join(data_dir, "correlations", "gene_dict_deepmind_protein_coding_new_protein_coding_08_04.pkl"), "wb") as f:
    pickle.dump(gene_dict_deepmind, f)