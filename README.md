# Cardiometabolic Risk Explorer

An interactive Streamlit dashboard for exploring how age, sex, lifestyle behaviors, and metabolic biomarkers relate to cardiometabolic disease risk and multimorbidity.
The tool enables dynamic filtering and visualization of NHANES-derived health data.

## Project Overview

This project builds a visualization tool using NHANES (National Health and Nutrition Examination Survey) data.
The dashboard supports exploratory analysis across three major areas:

### 1. Age Trends & Biomarker Correlations

- Smoothed biomarker trajectories across age

- Age-stratified biomarker correlation matrices

- Gender comparison panels

### 2. Biomarkers Across Disease Status & Age

- Distributional comparisons of biomarkers between disease groups

- Metabolic marker changes across comorbidity levels

- Identification of disease-specific metabolic signatures

### 3. Comorbidities & Lifestyle

- Summary statistics of comorbidity burden

- Biomarker variation with increasing comorbidity count

- Disease/lifestyle prevalence patterns by behavior levels (e.g., smoking, physical activity)

### Repository Structure

```
├── .devcontainer/               # Dev container setup files
├── 1_data_preprocessing.Rmd     # Data cleaning pipeline (R)
├── 1_data_preprocessing.html    # Knit HTML output
├── covariate_df.rData           # Intermediate processed data
├── processed_data.csv           # Final dataset for Streamlit app
├── questionnaire.csv            # NHANES questionnaire variable map
├── streamlit_app.py             # Main Streamlit dashboard
└── README.md                    # Documentation
```

### Dependencies
**Python**
```
pip install streamlit pandas numpy altair
```

**R Packages**
```
tidyverse
mice
haven
```

### How to Run the Dashboard

1. Clone this repository
```
git clone <your-repo-url>
cd <repo-folder>
```
2. Ensure the file **processed_data.csv** is available

3. Launch Streamlit:
```
streamlit run streamlit_app.py
```


### Key Features

- **Dynamic filtering** for demographics and lifestyle

- **Interactive visualizations** including boxplots, smoothed lines, correlation heatmaps

- **Gender-aware comparisons** across age, disease, and lifestyle

- **Comorbidity burden analysis** with summary metrics and visual displays


### Insights Enabled by This Tool

- Higher comorbidity counts are associated with higher BMI and waist circumference

- Disease groups such as diabetes and heart attack show elevated glucose and blood pressure

- Lifestyle behaviors (smoking, physical activity) display strong relationships with disease prevalence

- Biomarkers such as LDL and total cholesterol exhibit strong correlations

- Glucose and A1c cluster together, reflecting shared metabolic pathways

### Team Cntributions

**Kathy Liu**: Data cleaning, imputation pipeline, metabolic/comorbidity visualizations, gender encoding logic, overall integration for process book.
**Shu Tu**: contributed to the design and implementation of the age-stratified risk-factor visualizations and the biomarker correlation matrix. Explored multiple visualization approaches, refined the final designs. Overall integration for Streamlit App. 
**You Wu**: Explored multiple visualization approaches, implemented biomarker comparison across disease status, ages, and sexes, developed customizable thresholds based on careful research of clinical guidelines, identified future work directions for the project, overall integration for live demo.


