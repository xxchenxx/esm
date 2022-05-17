for i in $(seq 0 7); do
TORCH_HOME=/datadrive_c/yucheng/.cache CUDA_VISIBLE_DEVICES=${i} nohup python -u finetune_sup_head_full.py esm1b_t33_650M_UR50S /datadrive_c/yucheng/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --lr-factor 10 --split_file /datadrive_c/yucheng/clean_datasets/d1/d1_${i}.pkl --seed 1 --wandb-name 0516_d1_${i}_seed1_ep10_full_GPU${i} --epochs 10  > 0516_d1_${i}_seed1_ep10_full_GPU${i}.out &
done