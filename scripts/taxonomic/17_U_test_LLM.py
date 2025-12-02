#!/usr/bin/python3.5

# script name: U_test_LLM.py
# developed by: Nefeli Venetsianou
# description: 
    # Mann–Whitney U-tests to compare study distance distributions across relatedness categories ("none", "low", "medium", "high").
        # This test compares high VS rest (medium, low, none), medium VS rest, etc. 
        # Applied for both Taxonomic and Functional similarity runs. 
        # It adjusts p-values using the Benjamini–Hochberg FDR method and outputs the results as a TSV file.
# framework: CCMRI
# last update: 19/08/2025

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal

# Load your data
df = pd.read_csv("/ccmri/merged_total_studies_similarity_files/relatedness_with_distance_final_dedup.tsv", sep="\t")

# Storage for Kruskal-Wallis results
kw_results = []

def kruskal_test(df, sim_category, run_col):
    sub_df = df[df['SimCategory'] == sim_category]

    groups = []
    for cat in ['none', 'low', 'medium', 'high']:
        g = sub_df[sub_df[run_col] == cat]['Distance']
        if len(g) > 1:  # need at least 2 values
            groups.append(g)
    
    if len(groups) > 1:
        stat, p_val = kruskal(*groups)
        kw_results.append({
            'SimCategory': sim_category,
            'Run': run_col,
            'H_stat': stat,
            'p_value': p_val
        })
    else:
        kw_results.append({
            'SimCategory': sim_category,
            'Run': run_col,
            'H_stat': None,
            'p_value': None,
            'Note': 'Not enough groups'
        })

# Run for all runs and categories
for sim_cat in ['Taxonomic', 'Functional']:
    for run in ['run1', 'run2', 'run3']:
        kruskal_test(df, sim_cat, run)

# Save Kruskal-Wallis results
kw_results_df = pd.DataFrame(kw_results)
kw_results_df.to_csv("/ccmri/merged_total_studies_similarity_files/kruskal_results.tsv", sep="\t", index=False)

# Storage for Mann-Witney results
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

