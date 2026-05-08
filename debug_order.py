import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
print("Importing torch...")
import torch
print("torch ok")

print("Importing joblib...")
import joblib
print("joblib ok")

print("Importing pandas...")
import pandas as pd
print("pandas ok")

print("Loading crop model...")
joblib.load(os.path.join("app", "models", "crop_model.pkl"))
print("crop model loaded")

sys.exit(0)
