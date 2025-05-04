import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

data_path = os.path.join(project_root, 'data', 'Airlines_updated.csv')
df = pd.read_csv(data_path)

df['Delayed'] = df['Delay']

cat_cols = ['Airline', 'AirportFrom', 'AirportTo', 'Route', 'DayOfWeek']
num_cols = ['Time', 'Length']
features = num_cols + cat_cols
target = 'Delayed'

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", BinaryEncoder(), cat_cols)
])

X = df[features]
y = df[target]
preprocessor.fit(X)

X_processed = preprocessor.transform(X)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced_subsample',
    random_state=42,
    verbose=2
)
print("Fitting Model...")
rf.fit(X_processed, y,)

model_dir = os.path.join(project_root, 'models')
os.makedirs(model_dir, exist_ok=True)
joblib.dump(preprocessor, os.path.join(model_dir, 'rf_preprocessor.joblib'))
joblib.dump(rf, os.path.join(model_dir, 'rf_production_model.joblib'))

print("Production model and preprocessor saved.")