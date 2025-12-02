import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# === Load data ===
### Specific biome ###
file_dir = "/ccmri/similarity_metrics/data/algorithm_testing/21.07.2025_LLM_biome_test"
file_path = os.path.join(file_dir, "terrestrial_LLM_output.tsv")
dataset_name = os.path.basename(file_path).split("_LLM_output.tsv")[0]
### Total ###
#file_dir = "/ccmri/merged_total_studies_similarity_files"
#file_path = os.path.join(file_dir, "relatedness_with_distance_final_dedup.tsv")
#dataset_name = "Total"

df = pd.read_csv(file_path, sep="\t", header=None)
df.columns = ["Study_ID1", "Study_ID2", "Run1", "Run2", "Run3", "Distance", "SimilarityType"]

# === Reshape and clean ===
df_melted = df.melt(
    id_vars=["Study_ID1", "Study_ID2", "Distance", "SimilarityType"],
    value_vars=["Run1", "Run2", "Run3"],
    var_name="Run",
    value_name="Relatedness"
)

# Normalize relatedness values
df_melted["Relatedness"] = df_melted["Relatedness"].str.lower().str.strip()
df_melted["Relatedness"] = df_melted["Relatedness"].replace({
    "no": "none", "": "none", "na": "none"
})

# Clean numeric values
df_melted = df_melted[df_melted["Distance"] != "NA"]
df_melted["Distance"] = pd.to_numeric(df_melted["Distance"], errors="coerce")
df_clean = df_melted.dropna(subset=["Distance", "Relatedness", "SimilarityType"])

# Add similarity column
df_clean["Similarity"] = 1 - df_clean["Distance"]

# === Prepare output dir ===
output_dir = os.path.join(file_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# === Color palettes ===
taxonomic_palette = {
    "high": "#005a32",
    "medium": "#238b45",
    "low": "#41ab5d",
    "none": "#74c476"
}

functional_palette = {
    "high": "#4a1486",
    "medium": "#6a51a3",
    "low": "#807dba",
    "none": "#9e9ac8"
}

# === Combined violin plots (one plot per similarity type) ===
sns.set(style="whitegrid")

for sim_type in ["Taxonomic", "Functional"]:

    plt.figure(figsize=(10, 6))

    # Subset data for similarity type
    df_sub = df_clean[df_clean["SimilarityType"] == sim_type]

    if df_sub.empty:
        print(f"⚠️ Skipping empty similarity type: {sim_type}")
        continue

    # Choose palette
    if sim_type == "Taxonomic":
        palette = taxonomic_palette
    else:
        palette = functional_palette

    # === Main violin plot ===
    ax = sns.violinplot(
        data=df_sub,
        x="Relatedness",
        y="Similarity",
        order=["high", "medium", "low", "none"],
        palette=palette,
        inner=None,    # we’ll add medians manually
        cut=0
    )

    # === Add median lines for each relatedness group ===
    for rel in ["high", "medium", "low", "none"]:
        subset = df_sub[df_sub["Relatedness"] == rel]
        if subset.empty:
            continue
        median_val = subset["Similarity"].median()

        # Get x-position of the category
        xpos = ["high", "medium", "low", "none"].index(rel)

        # Draw median as a thick horizontal line
        plt.hlines(
            y=median_val,
            xmin=xpos - 0.35,
            xmax=xpos + 0.35,
            colors="black",
            linewidth=2
        )

    plt.title(f"{dataset_name} – {sim_type} LLM Relatedness", fontsize=14)
    plt.xlabel("Relatedness Level")
    plt.ylabel("Similarity")
    plt.ylim(0, 1)

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{dataset_name}_{sim_type.lower()}_combined_violins.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")
