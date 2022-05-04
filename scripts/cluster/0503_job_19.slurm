#!/bin/bash
# -----------------------------------------------------------------
#SBATCH -J finetune                 # Job name
#SBATCH -o finetune.%j.out          # Name of stdout output file
#SBATCH -e finetune.%j.err          # Name of stderr output file
#SBATCH -p rtx                        # Queue (partition) name
#SBATCH -N 1                          # Total # of nodes
#SBATCH -n 1                          # Total # of mpi tasks
#SBATCH -t 24:00:00                   # Run time (hh:mm:ss)
#SBATCH -A Deep-Protein                  # Project/Allocation name
# -----------------------------------------------------------------
module load cuda/11.3
module load cudnn/8.2
cd /work2/08090/tlc619/frontera/alphafold
source /home1/08090/tlc619/.bashrc
conda activate gcn
cd /work2/08090/tlc619/frontera/esm
git pull
python -u finetune_sup_head_dsee_parallel.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file S_target_classification.pkl --adv --gamma 3e-3 --steps 1 > 0503_job_19.out