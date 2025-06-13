import csv

# Step 1: Load the BioProject → (Study, Secondary Accession) mapping
bioproject_map = {}
with open('/ccmri/similarity_metrics/data/other/mgnify_studies_bioprojects.tsv', newline='') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        bioproject_map[row['BioProject']] = (row['Study'], row['Secondary Accession'])

# Step 2: Read matched_mgnify_spire.tsv and enrich each line
with open('/ccmri/similarity_metrics/data/other/matched_mgnify_spire.tsv', newline='') as infile, open('/ccmri/similarity_metrics/data/other/merged_output.tsv', 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter='\t')

    # Write header
    writer.writerow(['Sample', 'BioProject', 'Study', 'Secondary Accession', 'Biome'])

    # Process each line
    for row in reader:
        sample, bioproject, biome = row
        study, sec_acc = bioproject_map.get(bioproject, ('NA', 'NA'))
        writer.writerow([sample, bioproject, study, sec_acc, biome])
