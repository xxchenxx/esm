for i in $(seq 0 9); do
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-3 --split_file d1/d1_${i}.pkl --seed 1 --wandb-name 0513_d1_${i}  > 0513_d1_r_${i}.out
done
