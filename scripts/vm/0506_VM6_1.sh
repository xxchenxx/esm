CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target_classification.pkl --adv --gamma 1e-6 --steps 2 --wandb-name S_ds_adv_1e-6_step2 > 0506_S_g1e-6_step2_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target_classification.pkl --adv --gamma 1e-5 --steps 2 --wandb-name S_ds_adv_1e-5_step2 > 0506_S_g1e-5_step2_GPU1.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target_classification.pkl --adv --gamma 1e-4 --steps 2 --wandb-name S_ds_adv_1e-4_step2 > 0506_S_g1e-4_step2_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target_classification.pkl --adv --gamma 1e-3 --steps 2 --wandb-name S_ds_adv_1e-3_step2 > 0506_S_g1e-3_step2_GPU3.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target.pkl --adv --gamma 1e-6 --steps 2 --wandb-name Sr_ds_adv_1e-6_step2 > 0506_Sr_g1e-6_step2_GPU4.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target.pkl --adv --gamma 1e-5 --steps 2 --wandb-name Sr_ds_adv_1e-5_step2 > 0506_Sr_g1e-5_step2_GPU5.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target.pkl --adv --gamma 1e-4 --steps 2 --wandb-name Sr_ds_adv_1e-4_step2 > 0506_Sr_g1e-4_step2_GPU6.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target.pkl --adv --gamma 1e-3 --steps 2 --wandb-name Sr_ds_adv_1e-3_step2 > 0506_Sr_g1e-3_step2_GPU7.out &

