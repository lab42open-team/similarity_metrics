# CCMRI Study Similarity & LLM-based relatedness calculation
This repository contains scripts and related files for computing and processing similarity metrics, for both taxonomic anf functional data within the **CCMRI** framework.

## Overview
The scripts in this repository are used for:
- Filtering taxonomic & functional tables to a desired depth (e.g., genus level, frequency thresholds).
- Applying abundance and diversity thresholds to remove low-quality data.
- Preparing cleaned tables for downstream similarity metric calculations.
- Downsampling noise injection via multinomial distribution to check metrics robustness. 

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


