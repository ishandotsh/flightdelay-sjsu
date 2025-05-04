import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    classification_report
)

def load_data():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Airlines_updated.csv')
    df = pd.read_csv(data_path)
    
    # Prepare target variable
    df['Delayed'] = df['Delay']
    
    # Select features
    features = ['Flight', 'Time', 'Length', 'Airline', 'AirportFrom', 'AirportTo', 'Route', 'DayOfWeek']
    
    # Add additional features if needed
    if 'Airline_DelayRate' in df.columns:
        features.append('Airline_DelayRate')
    if 'Route_AvgDelay' in df.columns:
        features.append('Route_AvgDelay')
    
    # Prepare X and y
    X = df[features]
    y = df['Delayed']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate_model(model, X_test, y_test, model_name):
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_prob)
    }
    
    print(f"\n--- {model_name} Model Evaluation ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nDetailed Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return metrics

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Load models
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Load Random Forest model (new model)
    rf_model_path = os.path.join(models_dir, 'random_forest_model')
    rf_model = joblib.load(rf_model_path)
    
    # Load Logistic Regression model (old model)
    lr_model_path = os.path.join(models_dir, 'log_reg_acc_6549')
    lr_model = joblib.load(lr_model_path)
    
    # Evaluate models
    print("Comparing Model Performance:")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # Compare models
    print("\n--- Model Comparison ---")
    for metric in rf_metrics.keys():
        print(f"{metric}:")
        print(f"  Random Forest: {rf_metrics[metric]}")
        print(f"  Logistic Regression: {lr_metrics[metric]}")
        print(f"  Improvement: {((rf_metrics[metric] - lr_metrics[metric]) / lr_metrics[metric]) * 100:.2f}%\n")

if __name__ == '__main__':
    main()
