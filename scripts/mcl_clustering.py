#!/usr/bin/python3.5

# script name: mcl_clustering.py
# developed by: Nefeli Venetsianou
# description: 
    # MCL algorithm applied to cluster data.
    # Inflation parameter is usually 2.0 (typically in a range from 1.2 to 5). It can be changed:
        # higher infation parameter value --> more, smaller clusters
        # lower inflation parameter value --> less, larger clusters
# Input parameters: no input parameters are allowed for now. 
    # Script needs to be executed in Zorba HPC.
    # input_file and threshold must be defined for each run, according to input file. 
    # inflation parameter may be defined accordingly. 
# framework: CCMRI
# last update: 29/05/2024

import sys, re , os
import pandas as pd
import markov_clustering as mcl
import networkx as nx 
import matplotlib.pyplot as plt

parent_directory = "/ccmri/similarity_metrics/data/lf_super_table/phylum_filtered/" # filtered data according to taxonomic depth of choice directory 
output_directory = "/ccmri/similarity_metrics/data/lf_super_table/phylum_filtered/MCL_results/"

def read_file(filename):
    file_path = os.path.join(parent_directory, filename)
    # check is file exists and is a tsv file
    if not os.path.exists(file_path) or not file_path.endswith(".tsv"):
        print("File not found or not a .tsv file. Check parent directory '{}' for available input files.".format(parent_directory))
        sys.exit(1)
    return pd.read_csv(file_path, sep="\t")

def mcl_clustering(adj_matrix, inflation):
    # Run MCL algorithm
    result = mcl.run_mcl(adj_matrix, inflation=inflation)
    # retrieve clusters
    clusters = mcl.get_clusters(result)
    return clusters

def main():
    input_file = "v2.0_ph_filtered.tsv"
    df = read_file(input_file)
   # Define threshold according to ra descriptive stats (Q1 quartile)
    threshold = 2.428953e-04
    # Create graph
    G = nx.Graph()
    # Add edges to graph
    for _, row in df.iterrows():
        if row["Count"] >= threshold:
            G.add_edge(row["Taxa"], row["Sample"], weight=row["Count"])
    # Convert graph to an adjacency matrix
    adj_matrix = nx.to_numpy_array(G, weight="weight")
    # Perform MCL 
    clusters = mcl_clustering(adj_matrix, inflation=1.2)
    # Map index back to taxa and sample names
    index_to_names = {i:name for i, name in enumerate(G.nodes())}
    # Print clusters 
    for i, cluster in enumerate(clusters):
        cluster_names = [index_to_names[idx] for idx in cluster]
        print("Cluster {} : {}".format((i+1), cluster_names)) 
    # Adjusting output filename according to input filename 
    output_filename = re.sub(r"^lf_", "", input_file)[:-16] + "_mcl_clustering.png"
    output_file_path = os.path.join(output_directory, output_filename)
    # Visualize clusters
    pos = nx.spring_layout(G)
    colors = [plt.cm.tab20(float(i) / len(clusters)) for i in range(len(clusters))]
    plt.figure(figsize=(12,12))
    for cluster_id, cluster in enumerate(clusters):
        nx.draw_networkx_nodes(G, pos, nodelist=[index_to_names[idx] for idx in cluster], 
                               node_color=[colors[cluster_id]], label="Cluster {}".format(cluster_id + 1))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=2)
    plt.legend()
    # Adjusting title name by filename provided
    plot_title = re.sub(r"^lf_", "", input_file)[:-16] + " MCL Clustering Graph" 
    plt.title(plot_title)
    plt.savefig(output_file_path, format="png")
    
if __name__ == "__main__":
    main()