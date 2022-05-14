for i in $(seq 0 9); do
CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/d2/d2_${i}_classification.pkl --seed 1 --wandb-name 0514_d2_${i}_adv_5e-7_seed1_ep10_GPU5 --adv --gamma 5e-7 --epochs 10  > 0514_d2_${i}_adv_5e-7_seed1_ep10_GPU5.out
done