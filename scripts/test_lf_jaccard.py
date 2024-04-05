def lf_jaccard_score(input_file):
    # Dictionary to store counts for each sample
    sample_counts = {}
    # Record start time 
    start_time = time.time()
    
    with open(input_file, "r") as file:
        # Read header & sample IDs
        header = file.readline().strip().split("\t")
        #sample_ids = header[1:]
        # Initialize sample counts dictionary
        for sample_id in sample_ids:
            sample_counts[sample_id] = {}
        # Read counts for each taxon 
        for line in file:
            columns = line.strip().split("\t")
            taxon = columns[0]
            for i, count_str in enumerate(columns[2:]):
                # Adjust index to the count for taxon column
                count = float(count_str)
                sample_id = sample_ids[i]
                sample_counts[sample_id][taxon] = count
            #print(header, taxon, counts)
        # Initialize jaccard_score 
        jaccard_scores = []    
    # Iterate over indices of Sample IDs
    for i in range(len(sample_ids)):
    # Iterate over indices of sample IDs starting from i+1 to avoid comparing a sample to itself (duplicates)
        for j in range(i + 1, len(sample_ids)):
            # Define taxa sets of ith and jth sample 
            taxa_set1 = set(sample_counts[sample_ids[i]].keys())
            taxa_set2 = set(sample_counts[sample_ids[j]].keys())
            # Calculate intersection and union of taxa sets 
            intersection = taxa_set1.intersection(taxa_set2)
            union = taxa_set1.union(taxa_set2)
            # Calculate Jaccard similarity score
            jaccard_score = len(intersection) / len(union) 
            jaccard_scores.append((sample_ids[i], sample_ids[j], jaccard_score))
            #print("Jaccard Similarity Score between {}, {} = {}.".format(sample_ids[i], sample_ids[j], jaccard_distance))
    # Record end time 
    end_time = time.time()
    # Calculate execution time 
    execution_def_time = end_time - start_time
    logging.info("Execution def time = {} seconds".format(execution_def_time))
    return jaccard_scores