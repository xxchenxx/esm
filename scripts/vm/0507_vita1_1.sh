CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file S_target_classification.pkl --seed 3 --wandb-name S_ds_r16 > 0507_S_r16_seed3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file S_target_classification.pkl --seed 1 --wandb-name S_ds_r8_s16 --sparse 16  > 0507_S_r8_s16_seed1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file S_target_classification.pkl --seed 2 --wandb-name S_ds_r8_s16 --sparse 16  > 0507_S_r8_s16_seed2_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file S_target_classification.pkl --seed 3 --wandb-name S_ds_r8_s16 --sparse 16  > 0507_S_r8_s16_seed3_GPU3.out &
