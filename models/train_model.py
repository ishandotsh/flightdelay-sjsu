import os
import sys

def main():
    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)

    # Import required libraries
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from category_encoders import BinaryEncoder
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        classification_report, 
        roc_auc_score, 
        average_precision_score
    )
    import joblib

    # Load data
    def load_data():
        data_path = os.path.join(project_root, 'data', 'Airlines_updated.csv')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Use existing Delay column as Delayed
        df['Delayed'] = df['Delay']
        
        # Select relevant features
        # features = ['Flight', 'Time', 'Length', 'Airline', 'AirportFrom', 'AirportTo', 'Route', 'DayOfWeek']
        features = ['Time', 'Length', 'Airline', 'AirportFrom', 'AirportTo', 'Route', 'DayOfWeek']
        
        # Create a new dataframe with selected features and target
        processed_df = df[features + ['Delayed']]
        
        # Oversample minority class
        from sklearn.utils import resample
        
        # Separate majority and minority classes
        delayed_df = processed_df[processed_df['Delayed'] == 1]
        non_delayed_df = processed_df[processed_df['Delayed'] == 0]
        
        # Check if we have both classes
        if len(delayed_df) == 0 or len(non_delayed_df) == 0:
            print("Warning: Only one class present. Returning original data.")
            return processed_df
        
        # Upsample minority class
        if len(delayed_df) < len(non_delayed_df):
            minority_df = delayed_df
            majority_df = non_delayed_df
        else:
            minority_df = non_delayed_df
            majority_df = delayed_df
        
        # Upsample minority class
        minority_upsampled = resample(minority_df, 
                                      replace=True,     # sample with replacement
                                      n_samples=len(majority_df),    # to match majority class
                                      random_state=42)  # reproducible results
        
        # Combine majority class with upsampled minority class
        upsampled_df = pd.concat([majority_df, minority_upsampled])
        
        print("Class distribution after upsampling:")
        print(upsampled_df['Delayed'].value_counts(normalize=True))
        
        return upsampled_df

    # Preprocessing configuration
    cat_cols = ['Airline', 'AirportFrom', 'AirportTo', 'Route', 'DayOfWeek']
    # num_cols = ['Flight', 'Time', 'Length']
    num_cols = ['Time', 'Length']

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", BinaryEncoder(), cat_cols)
    ])

    # Data splitting function
    def prepare_data(df, test_size=0.2, random_state=42):
        X = df.drop('Delayed', axis=1)
        y = df['Delayed']
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Model Configurations
    models = {
        'KNN': Pipeline([
            ("pre", preprocessor),
            ("knn", KNeighborsClassifier(
                n_neighbors=7, 
                weights='distance',
                algorithm='auto'
            ))
        ]),
        'Decision Tree': Pipeline([
            ("pre", preprocessor),
            ("dt", DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced'
            ))
        ]),
        'Random Forest': Pipeline([
            ("pre", preprocessor),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced_subsample'
            ))
        ])
    }

    def evaluate_model(model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Handle binary and multiclass scenarios
        if y_prob.shape[1] > 1:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob.ravel()
        
        print(f"\n--- {model_name} Model Evaluation ---")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nROC AUC Score:")
        roc_auc = roc_auc_score(y_test, y_prob_pos)
        print(roc_auc)
        
        print("\nAverage Precision Score:")
        avg_precision = average_precision_score(y_test, y_prob_pos)
        print(avg_precision)
        
        return {
            'roc_auc': roc_auc,
            'avg_precision': avg_precision
        }

    # Main Execution
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    model_results = {}
    
    # Train and evaluate models
    for name, model in models.items():
        print(f"\nTraining {name} Model...")
        model.fit(X_train, y_train)
        model_results[name] = evaluate_model(model, X_test, y_test, name)
    
    # Select best model
    best_model_name = max(model_results, key=lambda k: model_results[k]['roc_auc'])
    best_model = models[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    
    # Save best model
    model_save_path = os.path.join(project_root, 'models', f'{best_model_name.lower().replace(" ", "_")}_model')
    joblib.dump(best_model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
