#!/usr/bin/python3.5

# script name: backtracking.py
# developed by: Nefeli Venetsianou
# description: aggregate counts according to child-parent relationships (based on gene_ontology_groups.tsv by Lars). 
# framework: CCMRI
# last update: 18/02/2025

import os
import time
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
    except Exception as e:
        logging.error(f"Error loading gene ontology file {e}")
        raise
    return go_dict

def load_input_data(input_file):
    """Loads input file (super table) into dictionary grouped by GO terms"""
    try:
        # Create nested dictionary where outter dic (data) is keyed by run and inner dic (default(int)) is keyed by go_term
        data = defaultdict(lambda: defaultdict(int)) 
        with open(input_file, "r") as f:
            next(f) # Skip header
            for line in f:
                go, run, count = line.strip().split()
                data[run][go] += int(count) # Aggregate count per GO term per Run
    except Exception as e:
        logging.error(f"Error loading input file {e}")
        raise
    return data

def aggregate_counts(data, go_dict):
    """Aggregate counts from children to parents per Run.
    This ensures thay parent GO terms include the counts of their children.
    """
    try:
        aggregated = defaultdict(lambda: defaultdict(int))
        def update_parent_counts(run, go_term, count):
            """Recursively update counts for parent GO_terms"""
            if go_term not in go_dict[go_term]:
                return # Stop if no parents (root node)
            for parent in go_dict[go_term]:
                aggregated[run][go_term] += count # Add child's count to parent
                update_parent_counts(run, parent, count)
        for run, go_terms in data.items():
            for go_term, count in go_terms.items():
                aggregated[run][go_term] += count # Retain original count
                update_parent_counts(run, go_term, count) # Aggregate count up the hierarchy
    except Exception as e:
        logging.error(f"Error in aggregating counts: {e}")
        raise
    return aggregated

def save_output(output_file, aggregated):
    """Saves the aggregated data in a tsv format."""
    try:
        with open(output_file, "w") as f:
            f.write("GO\tRun\tCount\n") # Write header
            for run, go_terms in aggregated.items():
                for go, count in go_terms.items():
                    f.write(f"{go}\t{run}\t{count}\n") # Write each record
    except Exception as e:
        logging.error(f"Error in saving output file: {e}")
        raise

def main():
    gene_ontology_file = "/ccmri/similarity_metrics/data/functional/gene_ontology_groups.tsv"
    input_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/raw_data/super_table/lf_v4.1_super_table.tsv"
    output_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/aggregated_lf_v4.1_super_table.tsv"
    start_time = time.time()
    go_dict = load_gene_ontology_data(gene_ontology_file)
    data = load_input_data(input_file)
    aggregated = aggregate_counts(data, go_dict)
    save_output(output_file, aggregated)
    end_time = time.time()
    logging.info(f"Total execution time is {end_time-start_time} sec.")

if __name__ == "__main__":
    main()