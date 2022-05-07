
CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_5_classification.pkl --adv --gamma 1e-5 > d1_5_advc.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_4_classification.pkl --adv --gamma 1e-5 > d1_4_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_3_classification.pkl --adv --gamma 1e-5 > d1_3_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_2_classification.pkl --adv --gamma 1e-5 > d1_2_advc.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_1_classification.pkl --adv --gamma 1e-5 > d1_1_advc.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_0_classification.pkl --adv --gamma 1e-5 > d1_0_advc.out &