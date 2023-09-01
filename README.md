# FT_S2A


# Data 

## Basenji2 data set

The data set used is the same as in [Kelley, PLOS Computational Biology, 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008050) and can be downloaded from [google cloud storage](https://console.cloud.google.com/storage/browser/basenji_barnyard/data).

You can convert the tensorflow dataloader to a pytorch dataloader using [save_basenji2_as_torch.py](https://github.com/Katharina782/FT_S2A/tree/master/enformer/save_basenji2_as_torch.py)

## snRNA-seq & snATAC-seq data

This data set was published by [Argelaguet et al., 2022](https://www.biorxiv.org/content/10.1101/2022.06.15.496239v1.full) and can be downloaded here: [10xmultiome](https://github.com/rargelaguet/mouse_organogenesis_10x_multiome_publication)

# Basenji2 in Pytorch 

## Pre-Training

### Basenji2'

This model [basenji_architecturel.py](https://github.com/Katharina782/FT_S2A/tree/master/basenji2_torch/basenji_architecture.py) has a bug, namely the activation function is applied before the residual connection. This leads to decreased performance. So far all fine-tuning experiments were performed with this model. 

To train the model use [basenji_training.py](https://github.com/Katharina782/FT_S2A/blob/master/basenji2_torch/basenji_training.py).


### Basenji2''

Here, I fixed the bug [basenji_architecture_res.py](https://github.com/Katharina782/FT_S2A/tree/master/basenji2_torch/basenji_architecture_res.py`) and apply the activation after the residual connection as described by [Kelley, PLOS Computational Biology, 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008050). 

To train the model use [basenji_training_res.py](https://github.com/Katharina782/FT_S2A/tree/master/basenji2_torch//basenji_training_res.py) or for multi-gpu training [basenji_training_res_multi_gpu.py](https://github.com/Katharina782/FT_S2A/tree/master/basenji2_torch//basenji_training_res_multi_gpu).

### Evaluations of models

To compute the correlations of gene expression predictions and observations at the gene TSS use [cage_correlations_basenji.py](https://github.com/Katharina782/FT_S2A/tree/master/basenji2_torch/cage_correlations_basenji.py).

Both Basenji2` and Basenji2`` are compared with each other, the original Enformer model and the small Enformer model here: 

* [comparison_correlation_across_positions.ipynb](https://github.com/Katharina782/FT_S2A/blob/master/enformer/comparison_correlation_across_positions.ipynb)
* [comparison_cage_correlation_tss.ipynb](https://github.com/Katharina782/FT_S2A/blob/master/enformer/comparison_cage_correlation_tss.ipynb)

# Fine-tuning


To finetune the original model use [finetuning_gastrulation.py](https://github.com/Katharina782/FT_S2A/tree/master/basenji2_torch/finetuning_gastrulation.py).

## Evaluations of models

To compute the correlations of gene expression predictions and observations at the gene TSS use `cage_correlations_basenji_finetune.py`.

The fine-tuned models are evaluated by computing correlation across all positions for ATAC-seq & RNA-seq and at the gene TSS for RNA-seq:

* For models fine-tuned just on ATAC-seq: [basenji_on_gastrulation_atac_finetune.ipynb](https://github.com/Katharina782/FT_S2A/tree/master/basenji2_torch/basenji_on_gastrulation_atac_finetune.ipynb)
* For models fine-tuned just on RNA-seq and jointly on RNA- & ATAC-seq: [basenji_on_gastrulation_join_finetune.ipynb](https://github.com/Katharina782/FT_S2A/tree/master/basenji2_torch/basenji_on_gastrulation_join_finetune.ipynb)

# Ridge regression predictions

A ridge regression model was fit to the orginal Basenji2 output tracks for mouse. The aim was to predict gene expression for mouse gastrulation cell types as a linear combination of the CAGE output tracks. [ridge_regression_encode.ipynb](https://github.com/Katharina782/FT_S2A/tree/master/basenji2_torch/ridge_regression_encode.ipynb)



# Small Enformers

The **MediumEnf** enformer architecture has five instead of eleven attention layers and no final pointwise convolution layer: [architecture_linear.py](https://github.com/Katharina782/FT_S2A/blob/master/enformer/architecture_linear.py)

The **SmallEnf** enformer architecture has five instead of eleven attention layers, no final pointwise convolution layer and no fully connected layer following the multi-head self-attention in every transformer block: [architecture_nolinear.py](https://github.com/Katharina782/FT_S2A/blob/master/enformer/architecture_nolinear.py)

## Training

Use the script [training_enformer.py](https://github.com/Katharina782/FT_S2A/blob/master/enformer/training_enformer.py) for training the smaller architectures. Make sure you import the correct architecture.

## Evaluation

Comparison of correlation across positions of the genome and at gene TSSs with the original Enformer and Basenji2 are found here: 

* [comparison_correlation_across_positions.ipynb](https://github.com/Katharina782/FT_S2A/blob/master/enformer/comparison_correlation_across_positions.ipynb)
* [comparison_cage_correlation_tss.ipynb](https://github.com/Katharina782/FT_S2A/blob/master/enformer/comparison_cage_correlation_tss.ipynb)

Enhancer-gene pair classification can be found here: [enhancer_gene_pair_classification.ipynb](https://github.com/Katharina782/FT_S2A/blob/master/enformer/enhancer_gene_pair_classification.ipynb)

