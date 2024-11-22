import pandas as pd
import numpy as np

import seaborn as sns


import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn import set_config
set_config(transform_output="pandas")

from utils import  extract_day_time_fe,  load_data


#------------------------------------

df = load_data(filepath = 'data/train.csv')


extract_day_time_fe(df, 'datetime')


X = df.drop(columns=["registered", "atemp"])

y = df["count"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_test = X_test.drop(columns=['count'])


# Calculate IQR and define bounds
q1 = X_train['count'].quantile(0.25)
q3 = X_train['count'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr) 
upper_bound = q3 + (1.5 * iqr) 

# Filter outliers
train_preprocessed = X_train.loc[(X_train['count'] >= lower_bound) & (X_train['count'] <= upper_bound)]

X_train = X_train.drop(columns=['count'])



def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x.to_numpy() / period * 2 * np.pi).reshape(-1, 1))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x.to_numpy() / period * 2 * np.pi).reshape(-1, 1))


num_pipe = make_pipeline(SimpleImputer(strategy='mean'),
                        MinMaxScaler(), PolynomialFeatures(include_bias=False,degree=2)
)

cat_pipe=make_pipeline(SimpleImputer(strategy='most_frequent'),
                         OneHotEncoder(handle_unknown='ignore', sparse=False)
                      )


class MyTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, column_names, period=1):
        self.column_names = column_names
        self.period =period
        
        
    def sin_transformer(self):
        # No need to call .to_numpy() because the input is already a NumPy array
        return FunctionTransformer(lambda x: np.sin(x / self.period * 2 * np.pi).reshape(-1, 1))

    def cos_transformer(self):
        # No need to call .to_numpy() because the input is already a NumPy array
        return FunctionTransformer(lambda x: np.cos(x / self.period * 2 * np.pi).reshape(-1, 1))


    def fit(self, X, y=None):
        return self  # The fit method typically does nothing for transformers
    
    def transform(self, X):
        X_transformed = X.copy()  
        for column_name in self.column_names:
            # Apply sine transformation directly to the column and store in the new column
            X_transformed[column_name + '_sin'] = self.sin_transformer().fit_transform(X[column_name].values.reshape(-1, 1)).reshape(-1)
            
            # Apply cosine transformation directly to the column and store in the new column
            X_transformed[column_name + '_cos'] = self.cos_transformer().fit_transform(X[column_name].values.reshape(-1, 1)).reshape(-1)
            
            # Example of another transformation - doubling the values in the column
            X_transformed[column_name] = X_transformed[column_name].apply(lambda x: x * 2)
        
        return X_transformed
   

feature_transform = ColumnTransformer(
    transformers=[
        ("num", num_pipe, ['temp', 'humidity', 'windspeed']),  # Apply num_pipe to numerical columns
        ("cat", cat_pipe, ['holiday', 'workingday', 'season', 'weather']),  # Apply cat_pipe to categorical columns
        ("custom", MyTransformer(column_names=["datetime_hour"], period=24), ["datetime_hour"])  # Apply MyTransformer to 'time' column
    ],
    remainder="drop"
)


feature_transform.fit(X_train)  

X_train_fe = feature_transform.transform(X_train)  
X_test_fe = feature_transform.transform(X_test)

X_train_fe.to_csv('./data/X_train_fe.csv', index=False)
X_test_fe.to_csv('./data/X_test_fe.csv', index=False)

np.save('./data/y_train.npy', y_train)
np.save('./data/y_test.npy', y_test)