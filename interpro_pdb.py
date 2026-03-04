#!/usr/bin/env python3

import requests
with open('accession.tsv', 'r') as file:
    for line in file:
        #protein_id = line.strip("\n")
        url = 'https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v6.pdb'
        try:
            # Copy a network object to a local file
            response = requests.get(url, stream=True)
            pdb_filename = url.split("/")[-1]
            with open(pdb_filename, mode="wb") as pdbfile:
                pdbfile.write(response.content)
        except Exception as e:
            print(f"An error occurred: {e}") #Write to a file the accesions that dont have a AF-PDB file
