# For specific MGYS
    # bring all MGYS with high similarity 

import os
import sys
import pandas as pd
import logging

# Load similarity files
def load_similarity_files(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["MGYS1", "MGYS2", "similarity"])
    return df

# Function to get high-similarity studies for a given MGYS
def get_high_similarity(df, mgys_id, threshold):
    results = df[((df["MGYS1"] == mgys_id) | df["MGYS2"] == mgys_id) & (df["similarity"] >= threshold)].copy()
    if results.empty:
        logging.warning("No data found.")
        return pd.DataFrame(columns=["Other", "similarity"])
    results["Other"] = results.apply(lambda row: row["MGYS2"] if row["MGYS1"] == mgys_id else row["MGYS1"], axis=1)
    return results[["Other", "similarity"]].sort_values(by="similarity", ascending=False)

def main():
    mgys_query = input("Enter MGYS ID to query: ").strip()
    threshold_input = input("Enter similarity threshold (default = 0.6): ").strip()
    threshold = float(threshold_input) if threshold_input else 0.6

    # Define directories and input files 
    taxonomic_parent_studies_similarity_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/similarity_with_studies/study_only_similarity/min_avg_distance"
    functional_parent_studies_similarity_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/post_pruned_distances/similarity_with_studies/studies_only_similarity/min_avg_distance"
    taxonomic_studies_similarity_file = os.path.join(taxonomic_parent_studies_similarity_dir, "merged_taxonomic_studies_similarity.tsv")
    functional_studies_similarity_file = os.path.join(functional_parent_studies_similarity_dir, "merged_functional_studies_similarity.tsv")
    results_parent_dir = "/ccmri/similarity_metrics/data/studies_similarity_tests"

    # Load files 
    taxonomic_df = load_similarity_files(taxonomic_studies_similarity_file)
    functional_df = load_similarity_files(functional_studies_similarity_file)

    # Get results 
    taxonomic_results = get_high_similarity(taxonomic_df, mgys_query, threshold)
    functional_results = get_high_similarity(functional_df, mgys_query, threshold)

    # Save outputs 
    taxonomic_outfile = os.path.join(results_parent_dir, f"{mgys_query}_taxonomic_similarities.tsv")
    functional_outfile = os.path.join(results_parent_dir, f"{mgys_query}_functional_similarities.tsv")
    taxonomic_results.to_csv(taxonomic_outfile, sep="\t", index=False)
    functional_results.to_csv(functional_outfile, sep="\t", index=False)

    logging.info(f"Results saved to: \n {taxonomic_outfile} \n {functional_outfile}")

if __name__ == "__main__":
    main() 
    

