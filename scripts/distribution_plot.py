#!/usr/bin/python3.5

# script name: distribution_plot.py
# developed by: Nefeli Venetsianou
# description: 
    # Plot distribution in taxonomic depth accross all samples per version.
    # Applied in normalized & aggregated data in Super Table (produced by: normalization_counts_ra.py & ra_aggregation.py)
# no input parameters are allowed (for now)
# framework: CCMRI
# last update: 14/05/2024

import pandas as pd
import numpy as np
import matplotlib 
# use non-interactive backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Parse data into a DataFrame 
df = pd.read_csv("/ccmri/similarity_metrics/data/super_table/long_format/lf_v1.0_super_table.tsv", sep="\t")
#print(df)

# Calculate frequency (number of occurrences) per unique taxon 
# value_counts() function in pandas returns a Series containing counts of unique values in a DF column. 
taxa_frequency = df["Taxa"].value_counts()

# Get descriptive statistics on taxa frequency 
taxa_description = taxa_frequency.describe()
print(taxa_description)

# Calculate lower quartile
lower_quartile = taxa_frequency.quantile(0.25)
med_quartile = taxa_frequency.quantile(0.50)
higher_quartile = taxa_frequency.quantile(0.75)
# Filter out frequencies below 
filtered_taxa_frequency = taxa_frequency[taxa_frequency >= med_quartile]
print("Number of filtered taxa: ", len(filtered_taxa_frequency))

# Plot histogram 
plt.figure(figsize=(20,15))
filtered_taxa_frequency.plot(kind="bar")
plt.xlabel("Taxa")
plt.ylabel("Total Count")
plt.title("Taxa Distribution Plot")
plt.xticks(rotation=90)
plt.yticks(range(0, max(filtered_taxa_frequency) + 1, 200))
plt.tight_layout()
plt.savefig("/ccmri/similarity_metrics/data/super_table/long_format/plots/distribution_test.png")


""" 
# Max - min taxa frequency
max_frequency = max(taxa_frequency)
min_frequency = min(taxa_frequency)
# Get corresponding taxa max - min frequency
taxa_max_frequency = taxa_frequency[taxa_frequency == max_frequency]
taxa_min_frequency = taxa_frequency[taxa_frequency == min_frequency]

# Print descriptive statistics
print("Max f:", max(taxa_frequency), "Taxon max f: ", taxa_max_frequency)
print("Min f:", min(taxa_frequency), "Taxon min f: ", taxa_min_frequency)
print("Mean f: ", mean(taxa_frequency))
print("Median f: ", median(taxa_frequency))
print("Variance f: ", variance(taxa_frequency))
print("Standard deviation f: ", stdev(taxa_frequency))

# Calculate mean, standard deviation
mean_freq = np.mean(taxa_frequency)
std_dev = np.std(taxa_frequency)
# Define threshold as percentage of mean (10% mean)
threshold = 0.05 * mean_freq
# Calculate z-score
z_scores = [(x - mean_freq)/std_dev for x in taxa_frequency]
# Find frequencies with z-scores extending threshold
outliers = [freq for freq, z_score in zip(taxa_frequency, z_scores) if z_score > threshold]
print(outliers, "Total outliers: ", len(outliers))  
"""
