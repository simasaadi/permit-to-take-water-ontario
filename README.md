# permit-to-take-water-ontario
Advanced analytics and dashboard on Ontario Permit to Take Water data
Ontario Permit-to-Take-Water (PTTW) Analytics

A full analytical exploration of Ontario’s Permit-to-Take-Water (PTTW) dataset, covering more than six decades of water-taking activity across the province.
The project includes data cleaning, feature engineering, exploratory analytics, advanced visualizations, and an interactive Streamlit dashboard.

Project Overview

This repository analyzes the PTTW dataset to understand trends in:

total permitted volumes over time

sector-specific water use

surface vs groundwater withdrawals

geographic patterns of high-volume permits

outliers and unusually large or long-duration permits

Three Jupyter notebooks walk through the full process:

01_data_cleaning.ipynb
Loads the raw dataset, standardizes column names, parses dates, computes daily and annual volumes, and generates key flags such as surface vs groundwater.

02_advanced_analysis.ipynb
Produces multi-angle analytics: time series, sector comparisons, volume distributions, duration analysis, municipality profiles, and high-volume permit identification.

03_spatial_mapping.ipynb
Generates spatial layers and point-based maps used in the dashboard, including quantile-based filters for high-volume withdrawals.

All outputs feed directly into the Streamlit dashboard.

Dashboard Features

The interactive app includes six analytical sections:

Overview – province-wide totals and time-series trends

Sector Explorer – how volumes vary across purpose categories

Surface vs Groundwater – comparative trends and distributions

Spatial Explorer – map of high-volume permits with filters

Outlier Explorer – large or long-duration permits

Data & Methodology – workflow and definitions

The dashboard is designed for clarity, exploration, and policy-relevant insight.

Repository Structure
├── data/
│   ├── raw/                       # raw PTTW dataset
│   ├── processed/                 # cleaned + engineered outputs
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_advanced_analysis.ipynb
│   ├── 03_spatial_mapping.ipynb
├── app.py                         # Streamlit dashboard
├── requirements.txt
└── README.md

Technologies

Python (Pandas, NumPy)

Plotly Express

GeoPandas

Streamlit

Folium / Mapbox tiles

Jupyter Notebooks

How to Run Locally

Clone the repo:

git clone https://github.com/yourusername/pttw-analytics.git
cd pttw-analytics


Install dependencies:

pip install -r requirements.txt


Start the dashboard:

streamlit run app.py
