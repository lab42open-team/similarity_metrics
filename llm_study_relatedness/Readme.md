# Pairwise Comparisons of MGnify Studies Using LLMs

This repository contains the scripts for performing **pairwise comparisons of metagenomic studies** using large language models (LLMs). The purpose is to leverage LLMs to capture semantic similarity between metagenomic studies using textual metadata. For each pair of studies, the model produces a similarity rating that reflects how “similar” the studies are based on their content. The levels of similarity are : [high,medium,low,no]

The workflow is organized around two main scripts:

1. **`pairwise_comparisons_LLM_V3_dynamic_process.sh`**  
   This Bash script orchestrates the workflow. It dynamically determines the number of parallel processes based on the size of the dataset, launches multiple invocations of the Python script "pairwise_comparisons_LLM_invoker_V2" in parallel, and merges partial outputs into final JSON results. It also logs runtime information, warnings, and any issues with missing or empty output files.

2. **`pairwise_comparisons_LLM_invoker_V2.py`**  
   This Python script is called by the Bash orchestrator. Each invocation processes a subset of study pairs for the specified LLM model and outputs a JSON file containing the similarity scores for those pairs. It receives as arguments the model to use, the dataset file, the study pairs file, the output filename, and process index information for parallel execution.

---

### Input files

1. **A list of all MGnify study IDs and their corresponding text**
   This input file (all_mgnify_studies.tsv) contains all the textual data coming from MGnify for each study ID.

3. **Available study pairs**
   This input file (merged_total_studies_similarity.tsv) contains all study pairs for which similarity scores were computed using similarity metrics. To compare the LLM-based results with the metric-based similarity scores, we need to identify exactly which MGnify study pairs have available metric scores. Because only MGnify versions 4.1 and 5.0 were used along with taxonomic and functional profiles, similarity scores are not available for all MGnify studies. Consequently, any study pairs that fall outside these versions do not have corresponding similarity metric values and cannot be included in the comparison.
