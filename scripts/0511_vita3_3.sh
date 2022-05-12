for i in $(seq 0 9); do
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-3 --rank 4 --lr-factor 1 --split_file d2/d2_${i}.pkl --seed 1 --wandb-name 0510_d2_r_${i}_adv_1e-6_seed1_GPU5_sap --adv --gamma 1e-6  > 0510_d2_r_${i}_adv_1e-6_seed1_GPU6_sap.out
done
