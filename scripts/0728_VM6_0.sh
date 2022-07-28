for i in $(seq 0 9); do
CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-3 --rank 4 --lr-factor 10 --split_file d2/d2_${i}_classification.pkl --seed 1 --wandb-name 0728_d2_${i}_mix_1e-6_seed1_GPU0_sap_0.2_0.1 --mix --mix_beta 0.1 > 0510_d2_${i}_mix_1e-6_seed1_GPU0_sap_0.2_0.1.out 
done