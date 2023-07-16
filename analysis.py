from Bio import pairwise2
import pandas as pd
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--input-csv", type=str)
parser.add_argument("--output-csv", type=str)
parser.add_argument("--num-sequences", type=int)
args = parser.parse_args()
data = pd.read_csv(args.input_csv)

label = data['label']

for i in range(args.num_sequences):
    scores = []
    output = data[f'generated_{i}']

    for j in range(len(label)):
        if isinstance(output[j], float) or isinstance(label[j], float):
            scores.append(0)
            continue
        result = pairwise2.align.globalms(label[j], output[j], 2, -1, -.5, -.1)[0]
        scores.append(result.score / (result.end - result.start))
    data[f'scores_generated_{i}'] = scores

data.to_csv(args.output_csv)