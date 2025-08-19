#!/usr/bin/python3.5

# script name: U_test_LLM.py
# developed by: Nefeli Venetsianou
# description: 
    # Mann–Whitney U-tests to compare study distance distributions across relatedness categories ("none", "low", "medium", "high").
        # Applied for both Taxonomic and Functional similarity runs. 
        # It adjusts p-values using the Benjamini–Hochberg FDR method and outputs the results as a TSV file.
# framework: CCMRI
# last update: 19/08/2025

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load your data
df = pd.read_csv("/ccmri/merged_total_studies_similarity_files/relatedness_with_distance_final_dedup.tsv", sep="\t")

# Storage for results
results = []

# Function to run U-tests: one category vs rest
def u_test_one_vs_rest(df, sim_category, run_col):    
    sub_df = df[df['SimCategory'] == sim_category]

    for cat in ['none', 'low', 'medium', 'high']:
        group_cat = sub_df[sub_df[run_col] == cat]['Distance']
        group_rest = sub_df[sub_df[run_col] != cat]['Distance']
        
        # Skip if not enough data
        if len(group_cat) < 2 or len(group_rest) < 2:
            results.append({
                'SimCategory': sim_category,
                'Run': run_col,
                'Category': cat,
                'n_cat': len(group_cat),
                'n_rest': len(group_rest),
                'U_stat': None,
                'p_value_raw': None,
                'Note': 'Not enough data'
            })
            continue
        
        u_stat, p_val = mannwhitneyu(group_cat, group_rest, alternative='two-sided')
        results.append({
            'SimCategory': sim_category,
            'Run': run_col,
            'Category': cat,
            'n_cat': len(group_cat),
            'n_rest': len(group_rest),
            'U_stat': u_stat,
            'p_value_raw': p_val,
            'Note': ''
        })

# Run once for Taxonomic, once for Functional based on run3
u_test_one_vs_rest(df, sim_category='Taxonomic', run_col='run1')
u_test_one_vs_rest(df, sim_category='Functional', run_col='run1')

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Adjust p-values for multiple testing (only where p-values exist)
mask = results_df['p_value_raw'].notna()
pvals = results_df.loc[mask, 'p_value_raw']

reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')  # Benjamini-Hochberg
results_df.loc[mask, 'p_value_adj'] = pvals_corrected
results_df.loc[mask, 'reject_H0'] = reject

# Save results
results_df.to_csv("/ccmri/merged_total_studies_similarity_files/run1_u_test_results.tsv", sep="\t", index=False)

print("Results saved with raw and adjusted p-values (FDR-BH).")

