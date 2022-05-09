CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_9_classification.pkl --adv > d2_9_advc.out 

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_8_classification.pkl --adv > d2_8_advc.out 

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_7_classification.pkl --adv > d2_7_advc.out 

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_6_classification.pkl --adv > d2_6_advc.out 

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_5_classification.pkl --adv > d2_5_advc.out 

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_4_classification.pkl --adv > d2_4_advc.out 

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_3_classification.pkl --adv > d2_3_advc.out 

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_2_classification.pkl --adv > d2_2_advc.out 

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_1_classification.pkl --adv > d2_1_advc.out 

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_0_classification.pkl --adv > d2_0_advc.out 