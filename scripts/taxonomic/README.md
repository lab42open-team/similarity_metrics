# Taxonomic scripts 
This folder  contains scripts for computing and analyzing similarity metrics based on **taxonomic profiles**, as well as for integrating and evaluating **LLM-based relatedness assessments** (for both taxonomic & functional data). The scripts support data preprocessing, similarity computation, ranking analysis, statistical testing, and visualization of both taxonomic/functional and LLM-derived results.
Scripts are organized with a numerical prefix that reflects the order they should be executed, and a desctiptive title that indicates the task performed (eg. 6_noise_injection.py).

## Scripts overview

- `1_isolate_abundances_file.py`
  Extracts and formats taxonomic abundance tables.

- `2_subset_creation.py`
  Creates study subsets used for downstream comparisons.

- `3_super_table_creation.py`
  Constructs the combined taxonomic feature table.

- `4_filter_taxonomy_depth.py`
  Filters taxonomic profiles by specific taxonomic level (e.g., genus).

- `5_normalize_data.py`
  Normalizes taxonomic abundance data based on relative abundances.

- `6_noise_injection.py`
  Injects noise (via multinomial distribution) into taxonomic profiles for robustness analysis.

- `7_sim_metrics_originalVSnoisy.py`
  Compares similarity metrics on original versus noisy taxonomic data.

- `8_ranking_metrics_performance.py`
  Assesses ranking performance of taxonomic similarity metrics.

- `9_avg_plot_ranking_performance_topK.py`
  Plots average ranking performance across top-K results.

- `9b_percentage_plot_ranking_performance_topK.py`
  Generates percentage-based performance plots.

- `10_sim_distances_batches_allVSall.py`
  Computes all-versus-all taxonomic study distances in batches.

- `11_LLM_output_clouds.sh`
  Shell script for aggregating LLM output files.

- `12_calculate_study_distance_matrix.py`
  Builds study-level distance matrices.

- `13_mined_info_retrieval.py`
  Retrieves mined metadata used for relatedness assessment.

- `14_match_relatedness.py`
  Matches LLM-based relatedness labels to taxonomic similarity results.

- `15_plot_matched_relatedness.py`
  Visualizes relationships between taxonomic similarity and LLM-based relatedness.

- `15b_plot_manual_vs_LLM_relatedness.py`
  Compares manually curated and LLM-derived relatedness labels.

- `16_plot_LLM_category_distribution_perBiomeIncluded.py`
  Plots the distribution of LLM relatedness categories across biomes.

- `16e_violin_plot_LLM_relatedness.py`
  Creates violin plots of LLM-based relatedness distributions.

- `17_u_test_LLM.py`
  Performs Mann–Whitney U tests on LLM-relatedness results.

- `17c_u_test_exclude_none_LLM.py`
  Repeats statistical tests excluding the “none” relatedness category.


## Notes 
- **Execution  order:** Scripts 1 to 5 handle data preprocessing and normalization. Script 6 performs noise injection for robustness evaluation. Scripts 7-10 compute similarity metrics and ranking analyses.
- **LLM-based relatedness analyses:** Scripts 11 to 17c implement LLM-based study relatedness workflows. These scripts use the same approach for both taxonomic and functional profiles, aggregating outputs, calculating study similarity matrices, matching with LLM-derived labels, visualizing outputs, and performing statistical tests. Please note that all LLM-based analyses for both taxonomic and functional datasets are handled in this folder, since the processing code is identical. 
- **Visualization:** Scripts 9, 15, 16, and 16e generate key performance and relatedness visualizations. They should be executed after the similarity calculations (7-10) and LLM-derived analyses (11-17c).
