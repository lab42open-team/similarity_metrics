# script name: plot_matched_relatedness.py
# developed by: Nefeli Venetsianou
# description: 
    # Plot matched manual check similarity with LLM assessed relatedness.
# framework: CCMRI
# last update: 23/07/2025

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Load the output file
parent_dir = "/ccmri/merged_total_studies_similarity_files"
df = pd.read_csv(os.path.join(parent_dir, "matched_relatedness_output.tsv"), sep='\t')

# Melt LLM runs for plotting
melted_llm = df.melt(
    id_vars=["Study1", "Study2"],
    value_vars=["run1", "run2", "run3"],
    var_name="Run",
    value_name="Label"
)

# Add manual answers as a separate 'Run' category
manual_df = df[["Manual_Answer"]].copy()
manual_df["Run"] = "manual"
manual_df.rename(columns={"Manual_Answer": "Label"}, inplace=True)

# Define the color palette
palette = {
    "run1": "#377eb8",
    "run2": "#e41a1c",
    "run3": "#4daf4a",
    "manual": "#984ea3"
}

# Enforce order of labels
label_order = ["none", "low", "medium", "high"]
melted_llm["Label"] = pd.Categorical(melted_llm["Label"], categories=label_order, ordered=True)
manual_df["Label"] = pd.Categorical(manual_df["Label"], categories=label_order, ordered=True)

# Plot 1: LLM + Manual
plt.figure(figsize=(8,5))
combined_df = pd.concat([melted_llm, manual_df[["Run", "Label"]]])
sns.countplot(data=combined_df, x="Label", hue="Run", palette=palette)
plt.title("Relevance Level Distribution: LLM Runs + Manual")
plt.xlabel("Relevance Level")
plt.ylabel("Count")
plt.legend(title="Source")
plt.tight_layout()
plt.savefig(os.path.join(parent_dir, "llm_manual_combined_plot.png"))
plt.close()

# Plot 2: LLM only
plt.figure(figsize=(8,5))
sns.countplot(data=melted_llm, x="Label", hue="Run", palette=palette)
plt.title("Relevance Level Distribution: LLM Runs Only")
plt.xlabel("Relevance Level")
plt.ylabel("Count")
plt.legend(title="Run")
plt.tight_layout()
plt.savefig(os.path.join(parent_dir, "llm_only_plot.png"))
plt.close()


# Basic Stats per run #
with open(os.path.join(parent_dir, "llm_exact_match_summary.txt"), "w") as f:
    total = len(df)
    for run in ['run1', 'run2', 'run3']:
        matches = (df[run] == df["Manual_Answer"]).sum()
        accuracy = matches / total * 100
        f.write(f"{run} exact matches: {matches} out of {total} ({accuracy:.2f}%)\n")

# Precision, Recall, F1-score #
labels = ["none", "low", "medium", "high"]
for run in ['run1', 'run2', 'run3']:
    print(f"\nClassification report for {run}:")
    report = classification_report(
        df['Manual_Answer'],
        df[run],
        labels=labels,
        zero_division=0,  # Avoid errors if some label missing in predictions
        output_dict=True
    )
    # Save output 
    with open(os.path.join(parent_dir, "llm_classification_reports.txt"), "w") as f:
        for run in ['run1', 'run2', 'run3']:
            f.write(f"Classification report for {run}:\n")
            report_str = classification_report(
                df['Manual_Answer'],
                df[run],
                labels=labels,
                zero_division=0  # avoid divide-by-zero warnings
            )
            f.write(report_str)
            f.write("\n" + "="*60 + "\n\n")

print("Plot saved as relevance_distribution_plot.png")
print("Exact match summary saved to llm_exact_match_summary.txt")
print("Classification reports saved to llm_classification_reports.txt")