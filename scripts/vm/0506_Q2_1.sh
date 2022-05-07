CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 3 --mix --wandb-name S_mix > 0506_S_mix_seed3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 1 --adv --gamma 1e-5 --wandb-name S_adv > 0506_S_adv_seed1_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 2 --adv --gamma 1e-5 --wandb-name S_adv > 0506_S_adv_seed2_GPU0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 3 --adv --gamma 1e-5 --wandb-name S_adv > 0506_S_adv_seed3_GPU0.out &
