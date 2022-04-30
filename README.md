# 
code explained:

the esm/sparse_multihead_attention.py contains a different version of multihead attention - it contains low-rank and sparse factors. 

# examples
## classification
python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S <fasta_root> sup --include mean per_tok --toks_per_batch 2048 --num_classes <num_classes> --idx <save_id> --lr 2e-2 --split_file <split_file> --mix 

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx m4 --lr 2e-2 --split_file S_target_4_classification.pkl --mix 