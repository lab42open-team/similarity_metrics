import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

class TrieNode:
    def __init__(self):
        # Each node has children and counts per sample
        self.children = defaultdict(TrieNode)
        # Store relative abundances per sample
        self.counts = defaultdict(float)

class Trie:
    def __init__(self):
        # Initialize the root as an instance of TrieNode
        self.root = TrieNode()
    
    def insert(self, taxon_path, sample, count):
        node = self.root
        for part in taxon_path.split(";"):
            node = node.children[part]
        node.counts[sample] += float(count)
    
    def get_samples(self):
        samples = set()
        def traverse(node):
            samples.update(node.counts.keys())
            for child in node.children.values():
                traverse(child)
        traverse(self.root)
        return samples
    
    def get_abundance_vector(self, sample, all_taxa):
        vector = []
        def traverse(node, path=""):
            if sample in node.counts:
                vector.append(node.counts[sample])
            else:
                vector.append(0.0)
            for part, child in node.children.items():
                traverse(child, path + ";" + part)
        for taxon in all_taxa:
            path_parts = taxon.split(";")
            current_node = self.root
            for part in path_parts:
                if part in current_node.children:
                    current_node = current_node.children[part]
                else:
                    current_node = None
                    break
            if current_node:
                traverse(current_node, taxon)
            else:
                vector.append(0.0)
        return vector
    
    def get_all_taxa(self):
        taxa = set()
        def traverse(node, path=""):
            if node.counts:
                taxa.add(path)
            for part, child in node.children.items():
                traverse(child, path + ";" + part)
        traverse(self.root)
        return taxa

def parse_tsv(file_path):
    """
    Parse the TSV file and build the Trie.
    """
    trie = Trie()
    with open(file_path, 'r') as file:
        next(file)  # Skip header
        for line in file:
            taxon, sample, count = line.strip().split('\t')
            trie.insert(taxon, sample, count)
    return trie

# Example Usage
file_path = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/down_filtered_data/filtered_v5.0_LSU_ge_filtered.tsv"
trie = parse_tsv(file_path)

# Get all unique taxa paths (for vector alignment)
all_taxa = trie.get_all_taxa()

# Get all unique samples
samples = list(trie.get_samples())

# Prepare a matrix to hold abundance vectors for all samples
abundance_matrix = []

# Retrieve the abundance vectors for all samples
for sample in samples:
    vector = trie.get_abundance_vector(sample, all_taxa)
    abundance_matrix.append(vector)

# Convert the list of vectors to a numpy array for efficient computation
abundance_matrix = np.array(abundance_matrix)

# Calculate pairwise cosine distances
cosine_dist_matrix = cosine_distances(abundance_matrix)

# Calculate pairwise Euclidean distances
euclidean_dist_matrix = euclidean_distances(abundance_matrix)

# Print the distance matrices
print("Cosine Distance Matrix:")
print(cosine_dist_matrix)

print("\nEuclidean Distance Matrix:")
print(euclidean_dist_matrix)
