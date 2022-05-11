for i in $(seq 0 9); do
CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_dsee_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 1e-3 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/d1/d1_${i}_classification.pkl --seed 1 --wandb-name 0510_d1_${i}_adv_1e-6_seed1_GPU5_sap --adv --gamma 1e-6  > 0510_d1_${i}_adv_1e-6_seed1_GPU5_sap.out
done
