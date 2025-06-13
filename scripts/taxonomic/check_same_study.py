# filter_biome_distances.py

def load_sample_to_study_map(mapping_file):
    sample_to_study = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            study_id, sample_id, *_ = line.strip().split()
            sample_to_study[sample_id] = study_id
    return sample_to_study

def filter_distances(distance_file, mapping_file, output_file):
    sample_to_study = load_sample_to_study_map(mapping_file)
    aquatic_samples = set(sample_to_study.keys())

    with open(distance_file, 'r') as infile, open(output_file, 'a') as outfile:
        header = infile.readline()
        outfile.write(header)
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            s1, s2, distance = parts
            if s1 in aquatic_samples and s2 in aquatic_samples:
                if sample_to_study[s1] != sample_to_study[s2]:
                    outfile.write(f"{s1}\t{s2}\t{distance}\n")

if __name__ == "__main__":
    filter_distances(
        "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v5.0_SSU_ge_filtered.tsv",
        "/ccmri/similarity_metrics/data/taxonomic/test_dataset/final_testing/terrestrial_studies.tsv",
        "/ccmri/similarity_metrics/data/taxonomic/test_dataset/final_testing/terrestrial_different_study_distances.tsv"
    )




"""
# top30_different_study.py

from collections import defaultdict

def load_sample_to_study_map(mapping_file):
    sample_to_study = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            study_id, sample_id = line.strip().split()
            sample_to_study[sample_id] = study_id
    return sample_to_study

def check_samples(data_file, mapping_file, output_file):
    sample_to_study = load_sample_to_study_map(mapping_file)
    all_distances = defaultdict(list)

    # Step 1: Collect all distances grouped by sample1
    with open(data_file, 'r') as infile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            sample1, sample2, similarity = parts
            try:
                similarity = float(similarity)
            except ValueError:
                continue
            all_distances[sample1].append((sample2, similarity))

    # Step 2: Sort and filter top 30 for each sample1
    with open(output_file, 'w') as outfile:
        outfile.write("Sample1\tSample2\tDistance\n")
        for sample1, neighbors in all_distances.items():
            sorted_neighbors = sorted(neighbors, key=lambda x: x[1])[:30]
            for sample2, similarity in sorted_neighbors:
                study1 = sample_to_study.get(sample1)
                study2 = sample_to_study.get(sample2)
                if not study1 or not study2 or study1 != study2:
                    outfile.write(f"{sample1}\t{sample2}\t{similarity}\n")


if __name__ == "__main__":
    data_file_aquatic = "/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/aquatic_test/aquatic_taxonomic_distances.tsv"
    data_file_terrestrial = "/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/terrestrial_test/terrestrial_taxonomic_distances.tsv"
    mapping_file_aquatic = "/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/aquatic_test/aquatic_sample_ids.tsv"
    mapping_file_terrestrial = "/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/terrestrial_test/terrestrial_samples_ids.tsv"
    output_aquatic = "/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/aquatic_test/top30_different_study.tsv"
    output_terrestrial = "/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/terrestrial_test/top30_different_study.tsv"
    check_samples(data_file_aquatic, mapping_file_aquatic, output_aquatic)
    print(f"Output saved to {output_aquatic}")
    check_samples(data_file_terrestrial, mapping_file_terrestrial, output_terrestrial)
    print(f"Output saved to {output_terrestrial}")
"""