# Tracing Inequitable Emissions in Global Trade using a Networks Approach

### Description
This folder contains Python tools and analysis scripts used in Kumar, N., Jensen, H.J., and Viegas,E. (2025). Tracing Inequitable Emissions in Global Trade using a Networks Approach. 

All data used in the paper are from publicly available sources. Data used in this paper is primarily from OECD's Inter-country Input Output Tables (ICIO) \url{https://www.oecd.org/sti/ind/inter-country-input-output-tables.htm} and \url{https://data-explorer.oecd.org/?tm=DF_ICIO_GHG_TRADE_2023}{OECD Greenhouse Gas Footprints (GHGFP)}. Per capita GDP data is retrieved from the World Bank from \url{https://data.worldbank.org/indicator/NY.GDP.PCAP.CD.}.Some data on the country level emissions is retrieved from the Emissions Database for Global Atmospheric Research (EDGAR) \url{https://edgar.jrc.ec.europa.eu/dataset_ghg70}. Raw data files are compressed and stored in `Data/Raw`. The ICIO data is named `{Year}_SML.csv` and the GHG flows are stored in `OECD_GHG.csv`. The datasets for the different sectoral classifications can be found with the subscripts `OECD_GHG_{sector}.csv`. The GDP per capita data from the World Bank Bank is stored in `gdp-per-capita-worldbank.csv`. 

### Contents
- `functions.py`: Core utility functions for data transformation and analysis.
- `trophic_tools.py`: Code forked from  (https://github.com/BazilSansom/Trophic-Analysis-Toolbox/tree/master). Tools for analyzing trophic-level data and ecological metrics relevant to trade-emissions studies with minor modifications for plots made with GDP percapita based ranking. See MacKay, Johnson & Sansom (2020), "How directed is a directed network", Royal Society Open Science, 7: 201138 doi: https://doi.org/10.1098/rsos.201138 for detailed description. 
- `Summary_plots.ipynb`: A Jupyter Notebook that produces summary plots produced for the analysis in the forthcoming paper.
- `Data_processing.ipynb`: A Jupyter Notebook with an example for generating the dataframes used in `Summary_plots.ipynb` from the raw data files described above.  
- `Strade_gravity.py`: Python code for the null-model inspired from Javier García-Algarra et al. (2019). jgalgarra/synthrade: Zenodo preserved release (V1.01). Zenodo. https://doi.org/10.5281/zenodo.2651147.
- `strade2_alphabeta.py`: Running monte-carlo simulations for multiple random realisations of the Strade_gravity.py model to produce results for the null model. 

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

