import pandas as pd

import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--input-csv", type=str)
parser.add_argument("--num-sequences", type=int)
args = parser.parse_args()

data = pd.read_csv(args.input_csv)
scores = []
for i in range(args.num_sequences):
    # print(f"scores_generated_{i}:", data[f'scores_generated_{i}'].mean())
    # print(f"scores_generated_{i} filtered:", data[f'scores_generated_{i}'][data[f'scores_generated_{i}'] > 0].mean())
    scores.append(data[f'scores_generated_{i}'].mean())

print(sum(scores) / len(scores))