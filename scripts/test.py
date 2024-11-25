from sklearn.metrics import adjusted_rand_score

# Example labels
true_labels = [0, 0, 1, 1, 2, 2]
predicted_labels = [0, 0, 1, 1, 2, 2]

# Calculate ARI
ari = adjusted_rand_score(true_labels, predicted_labels)
print("ARI (Example):", ari)
