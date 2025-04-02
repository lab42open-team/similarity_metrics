#!/usr/bin/python3.5

# script name: tree_creation.py
# developed by: Nefeli Venetsianou
# description: create tree according to child-parent relationships (based on gene_ontology_groups.tsv from Lars). 
# framework: CCMRI
# last update: 14/02/2025

import os 
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

def load_GO(ontology_file, output_file):
    # Dictionary to store parent nodes [keys] and child nodes [values]
    tree = defaultdict(set)
    # Set to track root nodes
    roots = set()
    # Load and read ontology file
    with open(ontology_file, "r") as f:
        for line in f:
            # Read each line and split to child - parent GO terms 
            child, parent = line.strip().split("\t")
            # Add child node under its parent 
            tree[parent].add(child)
            # If a node appears as a child, not a root
            roots.discard(child)
            # If parent not already in the tree, add it to roots (potentially root)
            if parent not in tree:
                roots.add(parent)
    # Save the first few entries of the tree to check
    print("\n🔹 First 10 Parent-Child Relationships in the Tree:")
    for i, (parent, children) in enumerate(tree.items()):
        print(f"{parent} -> {', '.join(children)}")
        if i == 9:  # Print only the first 10 entries
            break

    # Save all parent-child relationships to output file
    try:
        with open(output_file, "w") as f:
            f.write("🔹 All Parent-Child Relationships in the Tree:\n")
            for parent, children in tree.items():
                f.write(f"{parent} -> {', '.join(children)}\n")
        print(f"All relationships saved to {output_file}")
    except Exception as e:
        print(f"Error while saving tree: {e}")
                
    return tree, roots

def visualize_tree(tree, output_image):
    """
    Creates a directed graph visualization of the GO hierarchy and saves it as image.
    Args: 
        tree (dict): Dictionary mapping child GO terms to parent GO terms.
        output_image (str): Path to save visualization.
    """
    G = nx.DiGraph() # Create directed graph 
    # Add edges (relationships) to graph
    for parent, children in tree.items():
        for child in children:
            G.add_edge(parent, child)
    # Generate layout
    plt.figure(figsize=(12,8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="grey", node_size=2000, font_size=8)
    # Save Graph
    try:
        plt.savefig(output_image, dpi=300, bbox_inches="tight")
        print(f"Graph saved as {output_image}")
    except Exception as e:
        raise RuntimeError(f"Error while saving to {output_image}: {e}")
    plt.close()

def main():
    input_file = "/ccmri/similarity_metrics/data/functional/gene_ontology_groups.tsv"
    output_path = "/ccmri/similarity_metrics/data/functional/ontology_tree_test"
    tree_output_file = os.path.join(output_path, "GO_tree.txt")
    tree_image_file = os.path.join(output_path, "GO_tree.png")
    print("Loading gene ontology groups file...")
    tree, roots = load_GO(input_file, tree_output_file)
    print("Saving tree visualization...")
    visualize_tree(tree, tree_image_file)

if __name__ == "__main__":
    main()