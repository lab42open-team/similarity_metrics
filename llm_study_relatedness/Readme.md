# Pairwise Comparisons of Studies Using LLMs

This repository contains scripts for performing **pairwise comparisons of studies** using large language models (LLMs). The purpose is to go beyond simple metadata or sequence matching and leverage LLMs to capture semantic similarity between studies. For each pair of studies in a dataset, the model produces a similarity score that reflects how “related” the studies are based on their content.

The workflow is organized around two main scripts:

1. **`pairwise_comparisons_LLM_V3_dynamic_process.sh`**  
   This Bash script orchestrates the workflow. It dynamically determines the number of parallel processes based on the size of the dataset, launches multiple invocations of the Python script in parallel, and merges partial outputs into final JSON results. It also logs runtime information, warnings, and any issues with missing or empty output files.

2. **`pairwise_comparisons_LLM_invoker_V2.py`**  
   This Python script is called by the Bash orchestrator. Each invocation processes a subset of study pairs for a specific model and outputs a JSON file containing the similarity scores for those pairs. It receives as arguments the model to use, the dataset file, the study pairs file, the output filename, and process index information for parallel execution.

---

### How the workflow works

The Bash script first reads the dataset and the list of study pairs. Based on the dataset size, it decides how many parallel processes to run: smaller datasets might use only one or two processes, while larger datasets use multiple processes to speed up computation.  

For each model specified in the script, it launches the Python invoker multiple times in parallel. Each process calculates similarity scores for its assigned subset of study pairs using the specified LLM. Partial results are saved as JSON files with filenames like:

