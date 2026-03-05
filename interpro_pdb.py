#!/usr/bin/env python3

import requests
with open('accession.tsv', 'r') as file:
    for line in file:
        protein_id = line.strip("\n")
        url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v6.pdb"
        try:
            response = requests.get(url, stream=True)
            pdb_filename = url.split("/")[-1]
            with open(pdb_filename, mode="wb") as pdbfile:
                pdbfile.write(response.content)
            with open('interpro_pdb_downloaded.txt', 'a') as downloaded:
                donwloaded.write(protein_id + "\n")
        except Exception as e:
            with open('interpro_pdb_not_downloaded.txt', 'a') as not_downloaded:
                not_downloaded.write(protein_id  + "\n")
