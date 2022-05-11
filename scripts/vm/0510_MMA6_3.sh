for i in $(seq 0 9); do
CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-3  --split_file ~/clean_datasets/d2/d2_${i}_classification.pkl --seed 1 --wandb-name 0510_d2_classification_sap_seed1_GPU7_epoch6_1e-3 > 0510_d2_classification_sap_seed1_epoch6_1e-3_GPU7.out 
done