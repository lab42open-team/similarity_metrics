# Pairwise Comparisons of MGnify Studies Using LLMs

This repository contains the scripts for performing **pairwise comparisons of metagenomic studies** using Large Language Models (LLMs). The goal is to leverage LLMs to capture **semantic similarity** between metagenomic studies based on their textual metadata. For each pair of studies, the model produces a relatedness rating that reflects how similar the studies are in terms of content. The levels of relatedness are : **high,medium,low,no**

The workflow is organized around two main scripts:

1. **`pairwise_comparisons_LLM_V3_dynamic_process.sh`**  
   This Bash script orchestrates the workflow. It dynamically determines the number of parallel processes based on the size of the dataset, launches multiple invocations of the Python script `pairwise_comparisons_LLM_invoker_V2.py` in parallel, and merges partial outputs into final JSON results. It also logs runtime information, warnings, and any issues with missing or empty output files.

2. **`pairwise_comparisons_LLM_invoker_V2.py`**  
   This Python script is called by the Bash orchestrator. Each invocation processes a subset of study pairs for the specified LLM model and outputs a JSON file containing the similarity scores for those pairs. It receives as arguments the model to use, the dataset file, the study pairs file, the output filename, and process index information for parallel execution.

## Ollama Multi-Instance Configuration
- The script `pairwise_comparisons_LLM_invoker_V2.py` requires four concurrent Ollama instances by default, each listening on a distinct port (e.g., 11434, 11435, 11436, 11437).
- Users may modify the script to use different ports or fewer instances (e.g., a single instance), with a corresponding increase/decrease in execution time.

---

### Input files

1. **A list of all MGnify study IDs and their corresponding text**  
  This input file (`all_mgnify_studies.tsv`) contains all the textual data coming from MGnify for each study ID.

3. **Available study pairs**  
  This input file (`merged_total_studies_similarity.tsv`) contains all study pairs for which similarity scores were computed using cosine similarity. To compare the LLM-based results with the metric-based similarity scores, we first need to identify the MGnify study pairs for which metric scores are calculated. Similarity metrics are limited to MGnify versions 4.1 and 5.0 and to studies with available taxonomic or functional profiles. Therefore, any studies that fall outside there versions lack the corresponding metric similarity values and should be exluded from the comparison. 