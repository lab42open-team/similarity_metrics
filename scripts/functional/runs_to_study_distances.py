import pandas as pd
import numpy as np
import logging
from itertools import product

logging.basicConfig(level=logging.INFO)

def load_study_sample_mapping(mapping_file):
    """Loads the study-to-sample mapping from a TSV file."""
    try:
        study_sample_df = pd.read_csv(mapping_file, sep="\t")
        study_sample_dict = study_sample_df.groupby("MGYS_ID")["Sample_Name"].apply(set).to_dict()
        logging.info(f"Loaded study-sample mapping with {len(study_sample_dict)} studies.")
        return study_sample_dict
    except Exception as e:
        logging.error(f"Error loading study-sample mapping: {e}")
        raise

def load_sample_distance_matrix(distance_file):
    """Loads the sample-to-sample distance matrix from a TSV file."""
    try:
        df = pd.read_csv(distance_file, sep="\t")
        df["Run1"] = df["Run1"].astype("category")
        df["Run2"] = df["Run2"].astype("category")
        df["Distance"] = df["Distance"].astype("float64")
        logging.info(f"Loaded sample distance matrix with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading sample distance matrix: {e}")
        raise

def compute_study_distances(sample_distance_df, study_sample_dict):
    """Computes mean distances between studies more efficiently."""
    study_distances = []
    
    for study1, samples1 in study_sample_dict.items():
        for study2, samples2 in study_sample_dict.items():
            if (study1, study2) in study_distances:  # Skip duplicates
                continue

            filtered_df = sample_distance_df[sample_distance_df["Run1"].isin(samples1)]
            distances = filtered_df[filtered_df["Run2"].isin(samples2)]["Distance"]

            if not distances.empty:
                study_distances.append((study1, study2, distances.mean()))

    return pd.DataFrame(study_distances, columns=["Study1", "Study2", "Mean_Distance"])

def main():
    mapping_file = "/ccmri/similarity_metrics/data/functional/raw_data/merged_studies_samples.tsv"
    sample_distance_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/c_distances_batches_normalized_f_aggregated_0.75_0.01_lf_v5.0_super_table.tsv"
    output_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/study_distance_matrix_0.75_0.01_lf_v5.0.tsv"

    try:
        study_sample_dict = load_study_sample_mapping(mapping_file)
        sample_distance_df = load_sample_distance_matrix(sample_distance_file)
        study_distance_df = compute_study_distances(sample_distance_df, study_sample_dict)

        study_distance_df.to_csv(output_file, sep="\t", index=False)
        logging.info(f"Study distance matrix saved to {output_file}.")
    
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
