CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --wandb-name Sr_adv_1e-7_lr1e-3 --adv --gamma 1e-7  > 0513_Sr_adv_1e-7_seed1_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 --wandb-name Sr_adv_1e-7_lr1e-3 --adv --gamma 1e-7  > 0513_Sr_adv_1e-7_seed2_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 --wandb-name Sr_adv_1e-7_lr1e-3 --adv --gamma 1e-7  > 0513_Sr_adv_1e-7_seed3_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file ~/clean_datasets/S/S_target.pkl --seed 4 --wandb-name Sr_adv_1e-7_lr1e-3 --adv --gamma 1e-7  > 0513_Sr_adv_1e-7_seed4_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 1 --wandb-name S_adv_1e-7_lr1e-3 --adv --gamma 1e-7  > 0513_S_adv_1e-7_seed1_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 2 --wandb-name S_adv_1e-7_lr1e-3 --adv --gamma 1e-7  > 0513_S_adv_1e-7_seed2_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 3 --wandb-name S_adv_1e-7_lr1e-3 --adv --gamma 1e-7  > 0513_S_adv_1e-7_seed3_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 4 --wandb-name S_adv_1e-7_lr1e-3 --adv --gamma 1e-7  > 0513_S_adv_1e-7_seed4_GPU7.out &