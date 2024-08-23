import sys
import json
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.express as px

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(filepath):
    data = pd.read_csv(filepath)
    data.drop(columns=['Unnamed: 0'], inplace=True)
    data.drop_duplicates(inplace=True)
    return data

def preprocess_data(data):
    columns_with_na = ['Length (major axis)', 'Width (minor axis)', 'Thickness (depth)', 'Roundness', 'Aspect Ratio', 'Eccentricity']
    imputer = KNNImputer(n_neighbors=5)
    data[columns_with_na] = imputer.fit_transform(data[columns_with_na])
    label_encoder = LabelEncoder()
    data['Type'] = label_encoder.fit_transform(data['Type'])
    return data

def train_model(data, model_name):
    X = data.drop('Type', axis=1)
    y = data['Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
    if model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    elif model_name == 'LightGBM':
        model = lgb.LGBMClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    
    print(f"{model_name} Classifier")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        with open(config_file, 'r') as file:
            config = json.load(file)
        data = load_data('Almond.csv')
        data = preprocess_data(data)
        train_model(data, config['model_type'])
    else:
        print("No configuration file provided.")
