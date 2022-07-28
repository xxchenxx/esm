for i in $(seq 0 7); do 
CUDA_VISIBLE_DEVICES=${i} nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/d2/d2_${i}_classification.pkl --seed 2 --wandb-name 0516_d2_${i}_mix_seed2_ep10_GPU0 --mix --epochs 10  > 0516_d2_${i}_mix_seed2_ep10_GPU0.out & 
done