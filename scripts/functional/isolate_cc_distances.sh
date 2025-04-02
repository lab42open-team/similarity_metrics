# Retrieve distances for cc VS cc
awk 'NR==FNR {ids[$1]; next} $1 in ids && $2 in ids' cc_sec_access_sampleID_match.tsv /ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v4.1_SSU_ge_filtered.tsv >> cc_filtered_similarity_distances.tsv



# Retrieve distances for cc VS all
awk 'NR==FNR {ids[$1]; next} $1 in ids' cc_sec_access_sampleID_match.tsv /ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v4.1_LSU_ge_filtered.tsv > cc_VS_all_similarity_distances.tsv