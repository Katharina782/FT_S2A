
# usage examples
# python scripts/process/deconv_submit.py --output cellbender --model all --recompute True  --partition gpusaez
# python scripts/process/deconv_submit.py --output cellbender --model condition --recompute True --partition gpusaez
# python scripts/process/deconv_submit.py --output cellbender --model lesion_type --recompute True --partition gpusaez
# python scripts/process/deconv_submit.py --output cellranger --model all --recompute True --partition gpusaez
# python scripts/process/deconv_submit.py --output cellranger --model condition --recompute True --partition gpusaez
# python scripts/process/deconv_submit.py --output cellranger --model lesion_type --recompute True --partition gpusaez

import os
from pathlib import Path
import argparse
import subprocess

# get cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, required=True, help="output dir on cluster, e.g. /finetune/celltype")
parser.add_argument("--pretrained_model", type=str, required=True, help="path to pretrained model")
parser.add_argument("--optimizer", type=str, required=True, help="one of sgd or adam")
#parser.add_argument("--batch", type=int, required=False, default=16, help="batch size")
#parser.add_argument("--lr", type=float, required=False, default="False", help="")
parser.add_argument("--file_prefix", type=str, required=False, default="celltype", help="prefix of output files")
#parser.add_argument("--celltype", type=int, required=False, default="4", help="cell type to pick for single-cell finetuning")

args = parser.parse_args()

# check the arguments
data_dir = "/dkfz/cluster/gpu/data/OE0540/k552k/basenji/"
checkpoint_dir = args.checkpoint_dir
#print(checkpoint_dir)
pretrained_model = args.pretrained_model #"basenji_paper_param_real_data_no_it_corr_0.15_sgd_4_augmentTrue_model_validation_checkpoint"

python_script = "/home/k552k/katformer/Basenji2_torch/training_gastrulation_finetuning_both_single_cell_type_freezing.py"
prefix_file = args.file_prefix #"celltype"

# sbatch configs
MEM_GB = "30"
N_GPU = "1"
log_file ="single_celltype_freeze"
queue = "gpu"
#lr = args.lr #"0.0001"
opt = args.optimizer#"sgd"
#batch_size = args.batch
rna_tracks=37
atac_tracks=35
#celltype_ind = args.celltype 
#command_line_args = f"--lr {lr} --optimizer {opt} --augment -b {batch_size} --exp_rna {rna_tracks} --exp_atac {atac_tracks} --shuffle --celltype {celltype_ind} "


jobs = []


# submit GPU batch jobs for each sample
# how to deal with the right conda env?
for celltype in [4, 6]:
    for lr in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
        for batch_size in [4, 16]:
            print(f"submitting job with learning rate: {lr}, batch size: {batch_size}, celltype: {celltype}")
            command_line_args = f"--lr {lr} --optimizer {opt} --augment -b {batch_size} --exp_rna {rna_tracks} --exp_atac {atac_tracks} --shuffle --celltype {celltype} "
            command = f'bsub -R "rusage[mem={MEM_GB}G]" -gpu num={N_GPU}:j_exclusive=yes:gmem={MEM_GB}GB -o %J-{log_file}.out -e %J-{log_file}.err -J {log_file} -q {queue} source ~/.bash_profile && source ~/.bashrc && conda activate /omics/groups/OE0540/internal/users/mikulik/transformer && python {python_script} {command_line_args} {prefix_file} {data_dir} {checkpoint_dir} {pretrained_model}'
            print(command)
            #os.system(command)

            # Create a subprocess and append it to the list of jobs
            process = subprocess.Popen(
                ["bsub", "-R", f"rusage[mem={MEM_GB}G]", f"-gpu num={N_GPU}:j_exclusive=yes:gmem={MEM_GB}GB", "-o", f"%J-{log_file}.out", "-e", f"%J-{log_file}.err", "-J", log_file, "-q", queue, "bash", "-c", command]
            )
            jobs.append(process)