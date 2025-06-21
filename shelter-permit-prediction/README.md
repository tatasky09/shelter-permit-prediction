# Predicting Shelter Demand Using Permit & Housing Data in Seattle

**Overview:**  
Build a neighborhood-level predictive model to forecast monthly shelter usage, integrating Seattle building permits, census tract demographics, and shelter occupancy data. The goal: identify early risk signals to inform policy responses.

**Data:**  
- Seattle building permits (Residential & ADU): https://data.seattle.gov/resource/8tqq-u7ib.csv  
- King County homelessness metrics (HMIS dashboards)  
- ACS demographic data by census tract

**Tech Stack:**  
- Data wrangling: Python (`pandas`, `geopandas`), Socrata API  
- Modeling: scikit-learn (`RandomForestRegressor`, `XGBoost`), `SHAP` for explainability  
- Visualization: `folium` maps, `Plotly`, Tableau Public  
- (Optional) Scaling: Databricks Community Edition + PySpark

**Usage:**  
1. Run `notebooks/01-data_ingest.ipynb` to fetch and clean data.  
2. Use `src/modeling.py` to train/test the model:  
   ```python
   from src.modeling import train_model
   model, X_test, y_test = train_model(...)
   print(model.score(X_test, y_test))
