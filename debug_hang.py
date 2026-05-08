import _thread
import time
import sys

def kill():
    time.sleep(5)
    print("Interrupting main thread...", file=sys.stderr)
    _thread.interrupt_main()

_thread.start_new_thread(kill, ())

print("Importing pandas...")
import pandas
import os
print("Importing joblib...")
import joblib
print("Loading model...")
try:
    crop_model = joblib.load(os.path.join("app", "models", "crop_model.pkl"))
    print("Crop model loaded")
except Exception as e:
    print(e)

os.environ['MKL_NUM_THREADS'] = '1'
import torch
print("Done!")
