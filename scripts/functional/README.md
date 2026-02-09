# Functional scripts 
This folder contains scripts to compute, evaluate and benchmark similarity metrics based on functional profiles within the CCMRI framework. 
The workflow covers **data preprocessing, normalization, robustness testing, similarity computation, ranking analysis, and visualization.** 
Scripts are organized with a numerical prefix that reflects the order they should be executed, and a desctiptive title that indicates the task performed (eg. 9_remove_abundant_terms.py).

## Scripts overview

- `1_super_table_creation.py`  
  Constructs the functional feature table used for similarity computations.

- `2_backtracking.py`  
  Aggregate functional counts according to gene ontology (GO) child–parent relationships.

- `3_random_1000_runs_choice.py`  
  Downsample functional data to a random subset of 1000 runs for robustness and noise injection analyses.

- `4_noise_injection_fun.py`  
  Injects controlled noise into functional profiles for robustness testing.

- `5_sim_metrics_fun_originalVSnoisy.py`  
  Compares similarity metrics computed on original versus noise-perturbed functional data.

- `6_ranking_sim_metrics_performance.py`  
  Evaluates the ranking performance of different similarity metrics.

- `7_plot_avg_sim_metrics_performance_topK.py`  
  Plots average similarity metric performance across top-K rankings.

- `7b_percentage_plot_avg_sim_metrics_performance_topK.py`  
  Visualizes percentage-based performance summaries for similarity metrics.

- `8_filter_fun_data.py`  
  Filters functional GO term data based on Local and Global Frequency thresholds and generate TF-IDF rankings.

- `9_remove_redundant_terms.py`  
  Remove redundant GO terms from aggregated functional data using TF-IDF rankings and gene ontology parent-child relationships.

- `10_normalize_fun_data.py`  
  Normalizes functional profiles prior to similarity computation.

- `11_allVSall_sim_calculation.py`  
  Performs all-versus-all similarity calculations between studies.

- `12_retrieve_specific_distances.sh`  
  Shell script to extract specific similarity distance values from computed results.

- `13_runs_to_study_distances.py`  
  Aggregates multiple runs into final study-level distance summaries.


## Notes 
- **Execution  order:** Scripts 1 to 10 handle preprocessing and normalization, followed by 11 for similarity calculations, then 12-13 for post-processing and summarization.
- **LLM-based relatedness analyses:** After completing functional preprocessing and similarity calculations (scripts 1–13), LLM-based analyses are performed using the scripts in the `taxonomic` folder (scripts 11 to 17c). These scripts apply the same methodology to functional profiles, enabling consistent comparison across taxonomic and functional datasets.
- **Visualization:** Scripts 7 and 7b provide key performance visualizations and should be run after similarity and ranking analyses.


