import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load():
    return pd.read_csv("../data/Airlines_updated.csv")

def get_split(df):
    X = df.drop(['id', 'Delay'], axis=1)
    y = df['Delay']
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

def print_metrics(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def get_split_val(df, val_size=0.2, test_size=0.2):
    
    X = df.drop(['id', 'Delay'], axis=1)
    y = df['Delay']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=0,
        stratify=y
    )

    rel_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=rel_val_size,
        random_state=0,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
