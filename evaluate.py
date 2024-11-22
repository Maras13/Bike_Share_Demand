

import pandas as pd
import numpy as np

import seaborn as sns
sns.set_theme(style="ticks")

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.metrics import r2_score


import shap


# Extract model names and scores
model_names = list(results.keys())
scores = [results[model]["best_score"] for model in model_names]

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(model_names, scores, color='skyblue')
plt.xlabel('Score')
plt.title('Model Performance Comparison')
plt.xlim([0, 1.0])
plt.gca().invert_yaxis()  # Optional: invert y-axis for descending order
sns.despine()


shap_values_dict = {}

for model_name, model_info in results.items():
    best_model = model_info["best_model"]
    
    
    if isinstance(best_model, (RandomForestRegressor, 
                               HistGradientBoostingRegressor)):
                  
                  explainer = shap.Explainer(best_model, X_test_fe) 
                  shap_values = explainer.shap_values(X_test_fe, check_additivity=False)
                  
    else:
                   
                  explainer = shap.Explainer(best_model, X_test_fe) 
                  shap_values = explainer.shap_values(X_test_fe)
                  
                  
                  
                  
                  
    shap_values_dict[model_name] = shap_values


    shap_values_dict.keys()


    for model in shap_values_dict.keys():
    shap_values = shap_values_dict[model_name] 
    shap.summary_plot(shap_values, X_test_fe)



    for model in shap_values_dict.keys():
    shap_values = shap_values_dict[model_name] 
    shap.plots.force(explainer.expected_value, shap_values[0], X_test_fe.iloc[0, :], matplotlib = True)
    