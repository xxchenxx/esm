from struct import unpack
import pandas as pd
import requests
import os

def download_fasta_from_uniprot_id(data: str, out_file: str):

    temperatures = {}

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("Must provide a csv or pd.DataFrame with Protein_ID column.")

    df["uniprot_id"] = df['Protein_ID'].str.split('_').str[0] # fixing C0H3Q1_ytzI format to C0H3Q1

    temp_df = df[['uniprot_id', 'meltPoint']]
    
    mean_temp = temp_df.groupby(['uniprot_id'])[['meltPoint']].mean().add_suffix("_mean")
    mean_temp = mean_temp.reset_index(level=0)
    gene_uniprot_pairs = [
        row
        for row in mean_temp.itertuples(index=False, name="UniprotID")
    ]
    for idx in range(len(mean_temp)):
            uniprot, t = gene_uniprot_pairs[idx]
            if os.path.exists(f"meltome/{uniprot}.fasta"):
                temperatures[uniprot] = float(t)
            
    with open('meltome_temperature.csv', "w") as f:
        f.write("uniprot,temperature\n")
        for key in temperatures:
            f.write(f"{key},{temperatures[key]}\n")

    

if __name__ == "__main__":
    download_fasta_from_uniprot_id('cross-species.csv', 'fasta.fasta')