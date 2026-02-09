# CCMRI Study Similarity & LLM-based relatedness calculation
This repository contains scripts and related files for computing and processing similarity metrics, for both taxonomic anf functional data within the **CCMRI** framework.

## Overview
This repository is separated into the following folders:
- scripts:
    - intro_general_scripts: includes python and bash files to data preprocessing for data retrieval, preprocessing, and exploratory analyses that support the main similarity computations. These scripts were used to assemble study metadata, retrieve sample- and biome-level information from external resources, and generate summary statistics and visualizations. While not all scripts are required for reproducing the final similarity scores, they document the data preparation steps underlying the analysis. 
    - functional: this directory contains scripts used to compute, evaluate, and benchmark similarity metrics based on **functional profiles**. The workflow includes data normalization, noise injection, similarity computation, ranking performance analysis, and visualization of results. Several scripts were used for robustness analyses and sensitivity testing. 
    - taxonomic: This directory contains scripts for computing and analyzing similarity metrics based on **taxonomic profiles**, as well as for integrating and evaluating **LLM-based relatedness assessments** (for both taxonomic & functional data). The scripts support data preprocessing, similarity computation, ranking analysis, statistical testing, and visualization of both taxonomic/functional and LLM-derived results.
- llm_study_relatedness: this directory includes its own Readme.md file for futher explanation.

## Hardware requirements 
- Tested on Debian GNU/Linux 9.13 
- CPU execution on taxonomic/functional similarities calculations 
    - Intel(R) Xeon(R) Silver 4114 CPU (40 cores)
- GPU execution on LLM-based relatedness assessment
- Recommended RAM: >= 32GB
- Disk space: ~ 230 GB

## Software requirements
- Python 3.5+
- numpy
- pandas
- scipy
-scikit-learn
- matplotlib
- seaborn


