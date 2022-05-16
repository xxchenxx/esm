for i in $(seq 0 2); do
CUDA_VISIBLE_DEVICES=${i} nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/d2/d2_${i}.pkl --seed 2 --wandb-name 0515_d2_${i}_r_mix_seed2_ep10_GPU0 --mix --epochs 10  > 0515_d2_${i}_r_mix_seed2_ep10_GPU0.out &
done

for i in $(seq 4 7); do
CUDA_VISIBLE_DEVICES=${i} nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/d2/d2_${i}.pkl --seed 2 --wandb-name 0515_d2_${i}_r_mix_seed2_ep10_GPU0 --mix --epochs 10  > 0515_d2_${i}_r_mix_seed2_ep10_GPU0.out &
done