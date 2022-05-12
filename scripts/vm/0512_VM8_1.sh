CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --rank 16 --lr-factor 1 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --wandb-name 0512_Sr_ds_r16_s64_seed1_adv_1e-6_1e-3_GPU0 --adv --gamma 1e-6  > 0512_Sr_ds_r16_s64_seed1_adv_1e-6_1e-3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --rank 16 --lr-factor 1 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 --wandb-name 0512_Sr_ds_r16_s64_seed2_adv_1e-6_1e-3_GPU1 --adv --gamma 1e-6 > 0512_Sr_ds_r16_s64_seed2_adv_1e-6_1e-3_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --rank 16 --lr-factor 1 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 --wandb-name 0512_Sr_ds_r16_s64_seed3_adv_1e-6_1e-3_GPU2 --adv --gamma 1e-6 > 0512_Sr_ds_r16_s64_seed3_adv_1e-6_1e-3_GPU2.out &