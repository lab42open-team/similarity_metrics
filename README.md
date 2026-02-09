# CCMRI Study Similarity & LLM-based relatedness calculation
This repository contains scripts and resources for computing, analyzing, and visualizing study similarity metrics within the CCMRI framework, including both taxonomic and functional profiles, as well as LLM-based relatedness assessments.

## 1. Data 
Data were retrieved from MGnify using the API. Detailed information about the data download process and the analytical scripts used can be found in the repository "ccmri_data_gathering" here: https://github.com/lab42open-team/ccmri_data_gathering. 

## 2. Environment
Below is described the information on hardware and software used in this project. 

### Hardware requirements 
- Most of the analyses were conducted on Debian GNU/Linux 9.13, CPU execution (only), Intel(R) Xeon(R) Silver 4114 CPU (40 threads, 64GB RAM).
- Taxonomic and functional similarities calculations were conducted on Debian GNU/Linux 10, Intel(R) Xeon(R) CPU E5-2660 (20 cores, 1.5TB RAM).
- LLM-based relatedness was performed on GPU server (4×NVIDIA A100 GPU node).
- Disk space: ~ 230 GB

### Software requirements
- Python 3.5+
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn

## Setup
This repository includes an `environment.yml` file containing all the required dependencies.
To create and activate the Conda environment, run:

```bash
conda env create -f environment.yml
conda activate ccmri_env
```

## 3. Code
This repository is separated into the following folders:
- scripts:
    - intro_general_scripts: contains python and bash scripts for data retrieval, preprocessing and exploratory analyses, supporting the main similarity computations. These scripts implement standard **ETL (Extract, Transform, Load)** workflows to assemble study metadata, retrieve sample and biome-level information from external resources, and generate summary statistics and visualizations. While not all scripts are required for reproducing the final similarity scores, they document the data preparation steps and support **Explanatory Data Analysis (EDA)**.
    - functional: this directory contains scripts used to compute, evaluate, and benchmark similarity metrics based on **functional profiles**. The workflow includes data normalization, noise injection, similarity computation, ranking performance analysis, and visualization of results. Several scripts were used for robustness analyses and sensitivity testing. _The folder includes a README.md file with detailed description of each script._
    - taxonomic: this directory contains scripts for computing and analyzing similarity metrics based on **taxonomic profiles**, as well as for integrating and evaluating **LLM-based relatedness assessments** (for both taxonomic & functional data). The scripts support data preprocessing, similarity computation, ranking analysis, statistical testing, and visualization of both taxonomic/functional and LLM-derived results. _The folder includes a README.md file with detailed description of each script._
- llm_study_relatedness: this directory includes its own Readme.md file with detailed explanation for LLM-based study relatedness workflow.
