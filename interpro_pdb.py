#!/usr/bin/env python3

import requests

with open("accession.tsv", "r") as file:
    for line in file:
        protein_id = line.strip()
        if not protein_id:
            continue

        url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v6.pdb"
        pdb_filename = url.split("/")[-1]

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(pdb_filename, mode="wb") as pdbfile:
                pdbfile.write(response.content)

            with open("interpro_pdb_downloaded.txt", "a") as downloaded:
                downloaded.write(protein_id + "\n")
        except (requests.RequestException, OSError):
            with open("interpro_pdb_not_downloaded.txt", "a") as not_downloaded:
                not_downloaded.write(protein_id + "\n")
