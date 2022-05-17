CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_8_classification.pkl --seed 1 --wandb-name 0516_d2_8_mix_seed1_ep10_GPU0 --mix --epochs 10  > 0516_d2_8_mix_seed1_ep10_GPU0.out & 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_9_classification.pkl --seed 1 --wandb-name 0516_d2_9_mix_seed1_ep10_GPU0 --mix --epochs 10  > 0516_d2_9_mix_seed1_ep10_GPU3.out & 


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_8_classification.pkl --seed 2 --wandb-name 0516_d2_8_mix_seed2_ep10_GPU0 --mix --epochs 10  > 0516_d2_8_mix_seed2_ep10_GPU0.out & 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_9_classification.pkl --seed 2 --wandb-name 0516_d2_9_mix_seed2_ep10_GPU0 --mix --epochs 10  > 0516_d2_9_mix_seed2_ep10_GPU3.out & 




CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_6_classification.pkl --seed 1 --wandb-name 0516_d2_6_mix_seed1_ep10_GPU0 --mix --epochs 10  > 0516_d2_6_mix_seed1_ep10_GPU0.out & 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_7_classification.pkl --seed 1 --wandb-name 0516_d2_7_mix_seed1_ep10_GPU0 --mix --epochs 10  > 0516_d2_7_mix_seed1_ep10_GPU3.out & 

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_5_classification.pkl --seed 1 --wandb-name 0516_d2_5_mix_seed1_ep10_GPU0 --mix --epochs 10  > 0516_d2_5_mix_seed1_ep10_GPU0.out & 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_4_classification.pkl --seed 1 --wandb-name 0516_d2_4_mix_seed1_ep10_GPU0 --mix --epochs 10  > 0516_d2_4_mix_seed1_ep10_GPU3.out & 

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_3_classification.pkl --seed 1 --wandb-name 0516_d2_3_mix_seed1_ep10_GPU0 --mix --epochs 10  > 0516_d2_3_mix_seed1_ep10_GPU0.out & 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_2_classification.pkl --seed 1 --wandb-name 0516_d2_2_mix_seed1_ep10_GPU0 --mix --epochs 10  > 0516_d2_2_mix_seed1_ep10_GPU3.out & 


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_1_classification.pkl --seed 1 --wandb-name 0516_d2_1_mix_seed1_ep10_GPU0 --mix --epochs 10  > 0516_d2_1_mix_seed1_ep10_GPU0.out & 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_0_classification.pkl --seed 1 --wandb-name 0516_d2_0_mix_seed1_ep10_GPU0 --mix --epochs 10  > 0516_d2_0_mix_seed1_ep10_GPU3.out & 



CUDA_VISIBLE_DEVICES=3 python -u finetune_sup_head_full.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_0_classification.pkl --seed 1 --epochs 10  


CUDA_VISIBLE_DEVICES=3 python -u finetune_sup_head_regression_full.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_0.pkl --seed 1 --epochs 10  


for i in $(seq 0 7); do
CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_full.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/d2/d2_${i}.pkl --seed 1 --wandb-name 0516_d2_${i}_r_seed2_ep10_full_GPU${i} --epochs 10  > 0516_d2_${i}_r_seed2_ep10_full_GPU${i}.out &
done

for i in $(seq 0 7); do
CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_full.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file d2/d2_${i}.pkl --seed 1 --wandb-name 0516_d2_${i}_r_seed2_ep10_full_GPU${i} --epochs 10  > 0516_d2_${i}_r_seed2_ep10_full_GPU${i}.out 
done