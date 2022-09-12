import pandas as pd
import requests
import os

def download_fasta_from_uniprot_id(data: str, out_file: str):

    assert isinstance(data, str)


    assert isinstance(out_file, str)


    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("Must provide a csv or pd.DataFrame with Protein_ID column.")

    df["uniprot_id"] = df['Protein_ID'].str.split('_').str[0] # fixing C0H3Q1_ytzI format to C0H3Q1

    gene_uniprot_pairs = [
        row
        for row in df[["gene_name", "uniprot_id"]].itertuples(index=False, name="UniprotID")
    ]

    import pymp

    fasta_out = pymp.shared.list()
    http_err = pymp.shared.list()

    with pymp.Parallel() as p:
        for idx in p.range(len(gene_uniprot_pairs)):
            name, uniprot = gene_uniprot_pairs[idx]

            p.print(f"Thread {p.thread_num}: Downloading protein sequence: {name} ({uniprot})....")
            if pd.isna(uniprot):
                continue
            if os.path.exists(f"meltome/{uniprot}.fasta"):
                continue
            resp = requests.get(
                f"https://rest.uniprot.org/uniprotkb/search?&query=accession_id:{uniprot}&format=fasta"
            )
            if resp.ok:
                with open(f"meltome/{uniprot}.fasta", "w") as f:
                    f.write(resp.text)
  
            else:
                http_err.append(f"{name},{uniprot}")

    with open(out_file, "w") as f:
        f.writelines(fasta_out)


if __name__ == "__main__":
    download_fasta_from_uniprot_id('cross-species.csv', 'fasta.fasta')