import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print('Loading dataset...')
df = pd.read_csv(os.path.join('app', 'Data', 'crop_recommendation_extended.csv'))
print('Dataset loaded:', len(df), 'rows,', df['label'].nunique(), 'crops')
print('Crops:', sorted(df['label'].unique()))

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Training split done.')

print('Training RandomForestClassifier (n_jobs=1, n_estimators=100)...')
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
model.fit(X_train, y_train)
print('Training complete.')

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy:', round(acc * 100, 2), '%')

model_path = os.path.join('app', 'models', 'crop_model.pkl')
joblib.dump(model, model_path)
print('Model saved to', model_path)
print('DONE')
