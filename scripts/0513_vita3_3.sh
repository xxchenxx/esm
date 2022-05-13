for i in $(seq 0 9); do
CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-3 --split_file d2/d2_${i}.pkl --seed 1 --wandb-name 0513_d2_${i}  > 0513_d2_r_${i}.out
done
