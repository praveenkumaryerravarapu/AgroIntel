import sys
import time
import os
import threading

def kill_after(seconds):
    def _kill():
        time.sleep(seconds)
        print(f"\n[TIMEOUT] Hanging for >{seconds}s. Exiting...", file=sys.stderr)
        os._exit(1)
    threading.Thread(target=_kill, daemon=True).start()

kill_after(10)

print("Setting OpenMP/MKL flags...")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("Importing joblib...")
import joblib
print("Loading crop model...")
try:
    # Use mmap_mode to prevent loading the whole array into memory which might trigger computations
    crop_model_path = os.path.join("app", "models", "crop_model.pkl")
    crop_model = joblib.load(crop_model_path)
    print("[SUCCESS] Crop model loaded.")
except Exception as e:
    print(f"[ERROR] Crop model load failed: {e}")

print("Importing torch...")
import torch
print("[SUCCESS] torch imported.")

print("Loading disease model...")
try:
    disease_model_path = os.path.join("app", "models", "plant_disease_model.pth")
    # torch load might hang
    state = torch.load(disease_model_path, map_location='cpu', weights_only=True)
    print("[SUCCESS] Disease model state loaded.")
except Exception as e:
    print(f"[ERROR] Disease model load failed: {e}")

print("DONE.")
os._exit(0)
