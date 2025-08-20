import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# === Load data ===
file_dir = "/ccmri/similarity_metrics/data/algorithm_testing/21.07.2025_LLM_biome_test"
file_path = os.path.join(file_dir, "plants_LLM_output.tsv")
#file_dir = "/ccmri/merged_total_studies_similarity_files"
#file_path = os.path.join(file_dir, "relatedness_with_distance_final_dedup.tsv")
dataset_name = os.path.basename(file_path).split("_LLM_output.tsv")[0]
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

# === Plot for each SimilarityType & Relatedness level ===
sns.set(style="whitegrid")

for sim_type in ["Taxonomic", "Functional"]:
    for rel in ["high", "medium", "low", "none"]:
        subset = df_clean[(df_clean["SimilarityType"] == sim_type) & (df_clean["Relatedness"] == rel)]
        
        if subset.empty:
            print(f"⚠️ Skipping empty group: {sim_type} - {rel}")
            continue

        plt.figure(figsize=(8, 5))

        # Select color based on type and relatedness
        if sim_type == "Taxonomic":
            color = taxonomic_palette.get(rel, "gray")
        else:
            color = functional_palette.get(rel, "gray")

        sns.histplot(subset["Similarity"], kde=True, bins=30, color=color, kde_kws={"clip": (0, 1)})
        plt.title(f"{dataset_name.capitalize()} - {sim_type} - {rel.capitalize()} LLM Relatedness")
        #plt.xlim(0, subset["Distance"].max())  # Explicitly clip x-axis
        plt.xlabel("Similarity")
        plt.ylabel("No of studies' pairs")
        # Force y-axis ticks to be integers
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        #plt.ylim(bottom=0)
        plt.ylim(0, subset["Similarity"].value_counts().max() + 1)
        plt.xlim(left=0)
        # Clip max x to 1, but allow flexibility if your data max is smaller
        max_x = min(subset["Similarity"].max(), 1)
        plt.xlim(0, max_x)
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"{dataset_name}_{sim_type.lower()}_similarity_{rel}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
