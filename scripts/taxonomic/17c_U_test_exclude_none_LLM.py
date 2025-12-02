#!/usr/bin/python3.5

# script name: U_test_LLM_excluding_none.py
# developed by: Nefeli Venetsianou + updated by ChatGPT
# description:
#   Mann–Whitney U-tests comparing study distance distributions across 
#   relatedness categories ("low", "medium", "high"), EXCLUDING "none".
#   Tests are: high vs rest, medium vs rest, low vs rest.
#   Applied for both Taxonomic and Functional similarity runs.
#   Adjust p-values using Benjamini–Hochberg FDR.
# framework: CCMRI
# last update: 26/11/2025

import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests

# Load data
df = pd.read_csv(
    "/ccmri/merged_total_studies_similarity_files/relatedness_with_distance_final_dedup.tsv",
    sep="\t"
)

# Categories to test (exclude "none")
cats_to_use = ['low', 'medium', 'high']

# ------------------------------------------------------
# Part 1 — Kruskal-Wallis on high/medium/low only
# ------------------------------------------------------
kw_results = []

def kruskal_test(df, sim_category, run_col):
    sub_df = df[df['SimCategory'] == sim_category]
    sub_df = sub_df[sub_df[run_col].isin(cats_to_use)]   # EXCLUDE none

    groups = []
    for cat in cats_to_use:
        g = sub_df[sub_df[run_col] == cat]['Distance']
        if len(g) > 1:
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

# Run KW tests
for sim_cat in ['Taxonomic', 'Functional']:
    for run in ['run1', 'run2', 'run3']:
        kruskal_test(df, sim_cat, run)

# Save KW results
pd.DataFrame(kw_results).to_csv(
    "/ccmri/merged_total_studies_similarity_files/kruskal_excluding_none.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------
# Part 2 — Mann–Whitney one-vs-rest excluding "none"
# ------------------------------------------------------
results = []

def u_test_one_vs_rest(df, sim_category, run_col):
    sub_df = df[df['SimCategory'] == sim_category]
    sub_df = sub_df[sub_df[run_col].isin(cats_to_use)]   # EXCLUDE none

    for cat in cats_to_use:
        group_cat = sub_df[sub_df[run_col] == cat]['Distance']
        group_rest = sub_df[sub_df[run_col] != cat]['Distance']

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

        u_stat, p_val = mannwhitneyu(
            group_cat,
            group_rest,
            alternative='two-sided'
        )

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

# Run tests for both similarity types and all runs
for sim in ['Taxonomic', 'Functional']:
    for run in ['run1', 'run2', 'run3']:
        u_test_one_vs_rest(df, sim, run)

# Convert results
results_df = pd.DataFrame(results)

# Adjust p-values (BH-FDR)
mask = results_df['p_value_raw'].notna()
pvals = results_df.loc[mask, 'p_value_raw']

reject, pvals_corrected, _, _ = multipletests(
    pvals, method='fdr_bh'
)
results_df.loc[mask, 'p_value_adj'] = pvals_corrected
results_df.loc[mask, 'reject_H0'] = reject

# Save results
results_df.to_csv(
    "/ccmri/merged_total_studies_similarity_files/u_test_excluding_none.tsv",
    sep="\t",
    index=False
)

print("Tests excluding 'none' saved.")
