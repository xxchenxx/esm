CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 1 --wandb-name S > 0506_S_seed1_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 2 --wandb-name S > 0506_S_seed2_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 3 --wandb-name S > 0506_S_seed3_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 1 --noise --wandb-name S_noise > 0506_S_noise_seed1_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 2 --noise --wandb-name S_noise > 0506_S_noise_seed2_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 3 --noise --wandb-name S_noise > 0506_S_noise_seed3_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 1 --mix --wandb-name S_mix > 0506_S_mix_seed1_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 2 --mix --wandb-name S_mix > 0506_S_mix_seed2_GPU7.out &