# Gold Bioleaching — Kinetic Modelling + ML
 
Project for Principles of Extractive Metallurgy, 4th Sem MME, NITK Surathkal.
 
**Tejasvi Karanth | 241MT053**
 
---
 
## What it does
 
Takes lab data from a gold bioleaching experiment (*Pseudomonas fluorescens* in glycine medium) and runs two stages of analysis:
 
1. Fits a logistic kinetic model to each experimental time-series to extract E_max, k, and t_half
2. Trains ML regression models (Random Forest, Gradient Boosting, SVR) on those parameters to predict gold extraction under different conditions
 
Five process variables were studied: pH, inoculum concentration, pulp density, glycine concentration, and mineral-to-glycine ratio.
 
---
 
## How to run
 
```bash
pip install -r requirements.txt
python bioleaching_pipeline.py
```
 
Keep `RawData.xlsx` in the same folder. Figures are saved to `outputs/`.
 
---
 
## Files
 
```
├── bioleaching_pipeline.py   # main script
├── RawData.xlsx              # raw experimental data
├── requirements.txt
└── outputs/
    ├── 01_kinetic_fits.png
    ├── 02_parameter_sensitivity.png
    ├── 03_correlation_matrix.png
    ├── 04_model_comparison.png
    ├── 05_parity_plots.png
    ├── 06_shap_Emax.png
    ├── 07_response_surface.png
    └── ranked_conditions.csv
```
 
---
 
## Key result
 
Gradient Boosting predicted E_max (max gold extraction %) with LOO-CV R² = 0.81. Pulp density was the most important variable — lower pulp density consistently gave higher extraction.
