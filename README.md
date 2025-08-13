# CCMRI Similarity Metrics Pipeline
This repository contains scripts and related files for computing and processing similarity metrics, for both taxonomic anf functional data within the **CCMRI** framework.

## Overview
The scripts in this repository are used for:
- Filtering taxonomic & functional tables to a desired depth (e.g., genus level, frequency thresholds).
- Applying abundance and diversity thresholds to remove low-quality data.
- Preparing cleaned tables for downstream similarity metric calculations.
- Downsampling noise injection via multinomial distribution to check metrics robustness. 

## Requirements
- Python 3.5+  


