# Tracing Inequitable Emissions in Global Trade using a Networks Approach

### Description
This folder contains Python tools and analysis scripts used in Kumar, N., Jensen, H.J., and Viegas,E. (2025). Tracing Inequitable Emissions in Global Trade using a Networks Approach. 

### Contents

- `functions.py`: Core utility functions for data transformation and analysis.
- `trophic_tools.py`: Code forked from  (https://github.com/BazilSansom/Trophic-Analysis-Toolbox/tree/master). Tools for analyzing trophic-level data and ecological metrics relevant to trade-emissions studies with minor modifications for plots made with GDP percapita based ranking. See MacKay, Johnson & Sansom (2020), "How directed is a directed network", Royal Society Open Science, 7: 201138 doi: https://doi.org/10.1098/rsos.201138 for detailed description. 
- `Summary_plots.ipynb`: A Jupyter Notebook that produces summary plots produced for the analysis in the forthcoming paper.
- Strade_gravity.py: Python code for the null-model inspired from Javier García-Algarra, & Ghost Data Learner. (2019). jgalgarra/synthrade: Zenodo preserved release (V1.01). Zenodo. https://doi.org/10.5281/zenodo.2651147.
- strade2_alphabeta.py: Running monte-carlo simulations for multiple random realisations of the Strade_gravity.py model to produce results for the null model. 

### Usage

These tools are intended to be imported into larger workflows or run as part of exploratory analyses. The plotting notebook provides a visual overview of the results and is useful for reporting and verification.

### Requirements

Make sure the following Python packages are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `jupyter`
---

