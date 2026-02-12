# CCMRI Study Similarity & LLM-based relatedness calculation
This repository contains scripts and resources for computing, analyzing, and visualizing study similarity metrics within the CCMRI framework, including both taxonomic and functional profiles, as well as LLM-based relatedness assessments.

## 1. Data 
Data were retrieved from MGnify using the API. Detailed information about the data download process and the analytical scripts used can be found in the repository  **[`mgnify_data_retrieval`](https://github.com/lab42open-team/ccmri_public/tree/main/mgnify_data_retrieval)**.

## 2. Environment
Below is described the information on hardware and software used in this project. 

### Hardware requirements 
- Most of the analyses were conducted on Debian GNU/Linux 9.13, CPU execution (only), Intel(R) Xeon(R) Silver 4114 CPU (40 threads, 64GB RAM). The computational resources were provided by the Institute of Marine Biology, Biotechnology and Aquaculture of the Hellenic Centre for Marine Research.
- Taxonomic and functional similarities calculations were conducted on Debian GNU/Linux 10, Intel(R) Xeon(R) CPU E5-2660 (20 cores, 1.5TB RAM). The computational resources were provided by the Institute of Marine Biology, Biotechnology and Aquaculture of the Hellenic Centre for Marine Research.
- LLM-based relatedness was performed on GPU server (NVIDIA A100 GPU node). Access to GPU server for the LLM execution was provided by LifeWatch ERIC (European Research Infrastructure Consortium).
- Disk space: ~ 230 GB

### Software requirements
- Python >= 3.5
- numpy == 2.2.0
- pandas == 2.2.2
- scipy == 1.14.1
- scikit-learn == 1.6.0
- matplotlib == 3.9.2
- seaborn == 0.13.2
- Ollama == 0.6.8

### Setup
This repository includes an `environment.yml` file containing all the required dependencie for the pipeline, except for running the LLMs, which was performed on a GPU-server (information for running the LLMs is provided within the respective folder as described below).  
  To create and activate the Conda environment, run:

```bash
conda env create -f environment.yml
conda activate ccmri_env
```

## 3. Code
This repository is separated into the following folders:
- **[`scripts/`](scripts/)**  
  Main directory containing all analysis workflows.
    - **[`intro_general_scripts/`](scripts/intro_general_scripts/)** 
    This directory contains python and bash scripts for data retrieval, preprocessing and exploratory analyses, supporting the main similarity computations. These scripts implement standard **ETL (Extract, Transform, Load)** workflows to assemble study metadata, retrieve sample and biome-level information from external resources, and generate summary statistics and visualizations. While not all scripts are required for reproducing the final similarity scores, they document the data preparation steps and support **Explanatory Data Analysis (EDA)**.
    - **[`functional/`](scripts/functional/)** 
    This directory contains scripts used to compute, evaluate, and benchmark similarity metrics based on **functional profiles**. The workflow includes data normalization, noise injection, similarity computation, ranking performance analysis, and visualization of results. Several scripts were used for robustness analyses and sensitivity testing. See [`scripts/functional/README.md`](scripts/functional/README.md) for a detailed script-by-script description.
    - **[`taxonomic/`](scripts/taxonomic/)**
    This directory contains scripts for computing and analyzing similarity metrics based on **taxonomic profiles**, as well as for integrating and evaluating **LLM-based relatedness assessments** (for both taxonomic & functional data). The scripts support data preprocessing, similarity computation, ranking analysis, statistical testing, and visualization of both taxonomic/functional and LLM-derived results. See [`scripts/taxonomic/README.md`](scripts/taxonomic/README.md) for a detailed script-by-script description.
- **[`llm_study_relatedness/`](llm_study_relatedness/)**  
  This directory contains the complete workflow for **LLM-based study-relatedness analysis**. See [`llm_study_relatedness/Readme.md`](llm_study_relatedness/Readme.md).

## Citation
Venetsianou, N. K., Paragkamian, S., Kalaentzis, K., Loukas, A., Damianou, C., Lagani, V., Jensen, L. J., & Pafilis, E. (2026). _LLM-assessed relatedness of microbiome study descriptions aligns more strongly with functional than with taxonomic profile similarity_. (publication pending)

## License
BSD 2-Clause License

Copyright (c) 2026, Τhe CCMRI project, Hellenic Centre for Marine Research (HCMR). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.