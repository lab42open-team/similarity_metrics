### SIMPLE BARPLOT ###
"""
import matplotlib.pyplot as plt

# Load biome counts
biomes = []
counts = []

#with open("/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/biome_info/3b/biome_counts.tsv") as f: # taxonomic
with open("/ccmri/similarity_metrics/data/functional/raw_data/biome_info/biome_counts.tsv") as f: # functional

    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        biome, count = parts
        biomes.append(biome)
        counts.append(int(count))

# Plot
plt.figure(figsize=(14, 8))  # wider canvas
plt.bar(biomes, counts, color='skyblue')
plt.xlabel("Biome")
plt.ylabel("Number of Samples")
plt.xticks(rotation=90, ha='right')  # rotate and align
plt.tight_layout()

#plt.savefig("/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/biome_info/3b/biome_barplot.png", dpi=300) # taxonomic
plt.savefig("/ccmri/similarity_metrics/data/functional/raw_data/biome_info/biome_barplot.png", dpi=300) # functional

"""
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Step 1: Aggregate biomes to second level
biome_counts = defaultdict(int)
category_lookup = {}

#with open("/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/biome_info/3b/biome_counts.tsv") as f: # taxonomic
with open("/ccmri/similarity_metrics/data/functional/raw_data/biome_info/biome_counts.tsv") as f: # functional
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit("\t", 1)  # Split at the last tab character, separating biome and count
        if len(parts) != 2:
            continue  # Skip if the line doesn't have exactly two parts
        biome_full, count = parts
        count = int(count)  # Convert the count to an integer

        parts = biome_full.split(":")
        if len(parts) >= 3:
            category = parts[1]  # second level
            grouped = f"{parts[0]}:{parts[1]}"
        elif len(parts) == 2:
            category = parts[1]
            grouped = biome_full
        else:
            category = 'Unknown'
            grouped = biome_full

        biome_counts[grouped] += count
        category_lookup[grouped] = category

# Step 2: Assign a color automatically per category
unique_categories = sorted(set(category_lookup.values()))
cmap = cm.get_cmap('Set2', len(unique_categories))
category_to_color = {cat: mcolors.rgb2hex(cmap(i)) for i, cat in enumerate(unique_categories)}

# Step 3: Plot
sorted_items = sorted(biome_counts.items(), key=lambda x: x[1], reverse=True)
biomes, counts = zip(*sorted_items)
colors = [category_to_color[category_lookup[b]] for b in biomes]

plt.figure(figsize=(14, 8))
plt.bar(biomes, counts, color=colors)
plt.ylabel("Number of Samples")
plt.xticks(rotation=90)
plt.title("Sample Counts by Grouped Biome")
plt.tight_layout()

# Step 4: Legend
legend_handles = [mpatches.Patch(color=color, label=label) for label, color in category_to_color.items()]
plt.legend(handles=legend_handles, title="Category", loc="upper right")

# Save
#plt.savefig("/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/biome_info/3b/grouped_biome_barplot_auto.png", dpi=300) # taxonomic
plt.savefig("/ccmri/similarity_metrics/data/functional/raw_data/biome_info/grouped_biome_barplot_auto.png", dpi=300) # functional
