#!/usr/bin/python3.5

# script name: remove_redundant_terms.py
# developed by: Nefeli Venetsianou
# description: remove redundant term based on TF-IDF calculated and according to child-parent relationships (based on gene_ontology_groups.tsv by Lars). 
# framework: CCMRI
# last update: 11/04/2025

import os
import time
import pandas as pd
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO)


def load_gene_ontology_data(go_file):
    """Loads gene ontology relationshpis into a dictionary"""
    try:
        go_dict = defaultdict(set) # Use default dictionary for better handling of missing keys - initializes an empty set
        with open(go_file, "r") as f:
            for line in f:
                child, parent = line.strip().split()
                go_dict[child].add(parent)
        logging.info("Gene ontology fully-flatten tree loaded correctly.")
    except Exception as e:
        logging.error(f"Error loading gene ontology file {e}")
        raise
    return go_dict

def retrieve_all_ancestors(go_term, go_dict):
    """Recursively retrieve all ancestors of a GO-term"""
    try:
        ancestors = set()
        stack = list(go_dict.get(go_term, []))
        while stack:
            current = stack.pop()
            if current not in ancestors:
                ancestors.add(current)
                stack.extend(go_dict.get(current, []))
    except Exception as e:
        logging.error(f"Error in retrieving all ancestors: {e}")
        raise
    return ancestors

def prune_redundant_terms(input_file, go_dict, output_dir):
    """Prune redundant GO terms using TF-IDF ranking and GO clild-parent relationships."""
    try:
        df = pd.read_csv(input_file, sep="\t")
        df = df[["GO", "Run", "Ranking (TF-IDF)"]]
        logging.info(f"Input file loaded successfully. Head input file: {df.head()}")
        # Sort by TF-IDF score descending
        df_sorted = df.sort_values(by="Ranking (TF-IDF)", ascending=False)
        logging.info("DF sorted successfully.")
        # Initialize set of banned terms
        logging.info("Pruning process started...")
        banned_terms = set()
        final_rows = []
        for _, row in df_sorted.iterrows():
            go_term = row["GO"]
            if go_term in banned_terms:
                continue
            # Keep this GO term
            final_rows.append(row)
            # Ban all its ancestors 
            ancestors = retrieve_all_ancestors(go_term, go_dict)
            banned_terms.update(ancestors)
        pruned_df = pd.DataFrame(final_rows)
        output_filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, os.path.join(output_filename, "_tfidf_pruned.tsv"))
        pruned_df.to_csv(output_file, sep="\t", index=False)
        logging.info(f"Pruned files saved successfully to {output_file}")
        logging.info(f"GO terms reduces from {df["GO"].nunique()} to {pruned_df["GO"].nunique()}")
    except Exception as e:
        logging.error(f"Error in pruning redundant GO terms: {e}")
        raise
    return pruned_df

def main():
    go_file = "/ccmri/similarity_metrics/data/functional/gene_ontology_groups.tsv"
    input_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/filtered_cutOff/0.75_GL_0.01_LO/v4.1"
    input_file = os.path.join(input_dir, "aggregated_lf_v4.1_with_tfidf.tsv")
    output_dir = os.path.join(input_dir, "tfidf_pruned/")
    start_time = time.time()
    go_dict = load_gene_ontology_data(go_file)
    prune_redundant_terms(input_file, go_dict, output_dir)
    end_time = time.time()
    logging.info(f"Total execution time is {end_time-start_time} sec.")

if __name__ == "__main__":
    main()