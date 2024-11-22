import pandas as pd
import numpy as np

import seaborn as sns
sns.set_theme(style="ticks")

import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet



from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV
from sklearn import set_config
set_config(transform_output="pandas")


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from utils import  extract_day_time_fe,  load_data


import pickle


import mlflow
import mlflow.sklearn

import logging


#--------------------------------------

logging.basicConfig(level=logging.DEBUG)

mlflow.set_tracking_uri("file:./mlruns")


X_train_fe = pd.read_csv('./data/X_train_fe.csv')
X_test_fe = pd.read_csv('./data/X_test_fe.csv')

y_train = np.load('./data/y_train.npy')
y_test = np.load('./data/y_test.npy')

def train_model(X_train_fe, y_train, X_test_fe, y_test):
    
    model_params = {
        
        "LinearRegression": { 
            "model": LinearRegression(),
            "params": { 
            }
        
        },
                             
        "PoissonRegressor":{
            "model": PoissonRegressor(max_iter=1000),
            "params": {
                "alpha": [0.1, 0.5, 1.0, 5.0, 10.0]
            }
        },
            
        "Ridge": {
            "model": Ridge(max_iter=1000, tol=1e-3),
            "params": {
                "alpha": [0.1, 1.0, 10.0, 100.0]
            }
        },
        
        "Lasso": {
            "model": Lasso(max_iter=1000, tol=1e-3),
            "params": {
                
                "alpha":[0.001, 0.01, 0.1, 1.0, 10.0]
            }
        },
        
        "ElasticNet": {
            "model": ElasticNet(max_iter=1000, tol=1e-3),
            "params": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.1, 0.5, 0.7, 1.0]
            }
        },
             'HistGradientBoostingRegressor': {
            'model': HistGradientBoostingRegressor(),
            'params': {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10],
                'min_samples_leaf': [10, 20, 30]
            }
        },
        'RandomForestRegressor': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        }
    }
            
            
    
  
    
    results = {}
    mlflow.set_experiment("Model Training Experiment")
    
    for model_name, model_info in model_params.items():
         with mlflow.start_run(run_name=model_name):
        
            grid_search = GridSearchCV(model_info["model"], model_info["params"], cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train_fe, y_train)


            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
        
            results[model_name] = {
                "best_model": grid_search.best_estimator_,
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_
            }

              # Log metrics, parameters, and model to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric("best_r2_score", best_score)
            mlflow.sklearn.log_model(best_model, "model")

                   # Log additional metrics on test set
            y_test_pred = best_model.predict(X_test_fe)
            test_r2_score = r2_score(y_test, y_test_pred)
            mlflow.log_metric("test_r2_score", test_r2_score)


            with open('best_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)

        
        
        
    return results


results = train_model(X_train_fe, y_train, X_test_fe, y_test)

