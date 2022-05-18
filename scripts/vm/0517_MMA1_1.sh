for i in $(seq 0 9); do
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/d2/d2_${i}_classification.pkl --wandb-name 0517_d2_${i}_seed1_ep4_dsee_GPU${i} --rank 4 --seed 1 > 0517_d2_${i}_seed1_ep4_dsee_GPU${i}.out 
done
