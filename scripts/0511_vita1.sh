CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file S_target_classification.pkl --seed 1 > S_c_sap3d_s1.out & 
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file S_target_classification.pkl --seed 2 > S_c_sap3d_s2.out & 

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file S_target.pkl --seed 1 > S_r_sap3d_s1.out & 
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file S_target.pkl --seed 2 > S_r_sap3d_s2.out & 



CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file S_target_classification.pkl --seed 1 > S_c_sap3d_s1_1e-3.out & 
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file S_target_classification.pkl --seed 2 > S_c_sap3d_s2_1e-3.out & 

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file S_target.pkl --seed 1 > S_r_sap3d_s1_1e-3.out & 
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file S_target.pkl --seed 2 > S_r_sap3d_s2_1e-3.out & 



CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file S_target_classification.pkl --seed 3 > S_c_sap3d_s3_1e-3.out & 
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file S_target_classification.pkl --seed 4 > S_c_sap3d_s4_1e-3.out & 

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file S_target.pkl --seed 3 > S_r_sap3d_s3_1e-3.out & 
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --split_file S_target.pkl --seed 4 > S_r_sap3d_s4_1e-3.out & 




CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file S_target_classification.pkl --seed 1 > S_c_sap3d_s1.out & 
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file S_target_classification.pkl --seed 2 > S_c_sap3d_s2.out & 
CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file S_target_classification.pkl --seed 1 > S_c_sap3d_s1.out & 
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file S_target_classification.pkl --seed 2 > S_c_sap3d_s2.out & 