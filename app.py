from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

from preprocess import MyTransformer

from utils import extract_day_time_fe




app = Flask(__name__)

model = joblib.load("best_model.pkl")
feature_transform = joblib.load("feature_transform.pkl")




@app.route('/')
def home():
    return "Bike Share Demand Prediction API is running."



@app.route('/predict')
def predict():
    df = pd.read_csv("./data/test.csv")

    extract_day_time_fe(df, 'datetime')
            

    transformed_data = feature_transform.transform(df)
            

    predictions = model.predict(transformed_data)


    predictions_df = pd.DataFrame({'Predictions': predictions})


    return predictions_df.to_string(index=False)
  
    
        
    
if __name__ == '__main__':
    app.run(debug=True)



