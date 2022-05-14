for i in $(seq 0 9); do
CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/d2/d2_${i}.pkl --seed 1 --wandb-name 0514_d2_${i}_r_adv_5e-6_seed1_ep10_GPU0 --adv --gamma 5e-6 --epochs 10  > 0514_d2_${i}_r_adv_5e-6_seed1_ep10_GPU0.out
done