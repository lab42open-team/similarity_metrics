import pandas as pd
from scipy.stats import mannwhitneyu

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
                'p_value': None,
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
            'p_value': p_val,
            'Note': ''
        })

# Run once for Taxonomic, once for Functional based on runX
u_test_one_vs_rest(df, sim_category='Taxonomic', run_col='run3')
u_test_one_vs_rest(df, sim_category='Functional', run_col='run3')

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("/ccmri/merged_total_studies_similarity_files/run3_u_test_results.tsv", sep="\t", index=False)

print("✅ Results saved to 'run3_u_test_results.tsv'")
