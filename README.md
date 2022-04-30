# 
code explained:

the esm/sparse_multihead_attention.py contains a different version of multihead attention - it contains low-rank and sparse factors. 

# examples
## classification
Usage:
python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S <fasta_root> sup --include mean per_tok --num_classes <num_classes> --idx <save_id> --lr 2e-2 --split_file <split_file> (Optional: --mix)

Examples:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --num_classes 5 --idx S_4 --lr 2e-2 --split_file S_target_4_classification.pkl --mix > S_4_dsee_classification.out & # low-rank + sparse for classification 

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --idx S_4 --lr 2e-2 --split_file S_target_4.pkl --mix > S_4_dsee_regression.out & # low-rank + sparse for regression 

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --num_classes 5 --idx S_4 --lr 2e-2 --split_file S_target_4_classification.pkl --mix > S_4_head_classification.out & # head tuning for classification

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --idx S_4 --lr 2e-2 --split_file S_target_4.pkl --mix > S_4_head_regression.out &  # head tuning for regression
````

Change --idx to have different saved model. 