#!/usr/bin/python3.5

# script name: plot_GO_terms_counts.py
# developed by: Nefeli Venetsianou
# description: Bar-plots of GO terms according to counts. 
# framework: CCMRI
# last update: 25/02/2025

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/filtered_cutOff/test_set_logs"
input_file = os.path.join(file_path, "full_description_ERR9456921_filtered_0.75_0.01_log.tsv")
output_path = os.path.join(file_path, "plots")
output_file = os.path.join(output_path, "barplot_{}.png".format(os.path.basename(input_file)[:-4]))
df = pd.read_csv(input_file, sep="\t")
print(df.head())

# Drop entity type -24 (unused)
df = df[df["Entity_Type"].isin([-21,-22,-23])]

# Replace Entity_Type IDs with category names
category_mapping = {-21:"Biological Process", -22:"Cellular Component", -23:"Molecular Function"}
df["Category"] = df["Entity_Type"].map(category_mapping)

# Extract only first element in full name (until first comma)
df["Short_Name"] = df["Full_Name"].apply(lambda x: x.split(",")[0] if pd.notna(x) else x)

# Set up figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18,6), sharex=False, sharey=False)

# Define colors for each category 
colors = {"Biological Process":"lightblue", "Cellular Component":"orange", "Molecular Function":"lightgreen"}

# Iterate through categories and plot them in subplots
for ax, (category, color) in zip(axes, colors.items()):
    cat_df = df[df["Category"] == category].nlargest(30, "Count") # Top 30 terms
    sns.barplot(y=cat_df["Short_Name"], x=cat_df["Count"], ax=ax, color=color)
    ax.set_title(f"{category}\nTotal: {len(cat_df)} terms.")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig(output_file)
print(f"Output saved to: {output_file}")
