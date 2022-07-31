for i in $(seq 0 9); do
TORCH_HOME=/datadrive_c/yucheng/.cache CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee_sap.py esm1b_t33_650M_UR50S /datadrive_c/yucheng/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 1e-3 --rank 4 --lr-factor 10 --split_file /datadrive_c/yucheng/clean_datasets/d1/d1_${i}_classification.pkl --seed 1 --wandb-name 0730_d1_${i}_mix_1e-6_seed1_GPU3_sap_0.5_0.2 --mix --mix-alpha 0.5 > 0730_d1_${i}_mix_1e-6_seed1_GPU3_sap_0.5_0.2.out 
done
