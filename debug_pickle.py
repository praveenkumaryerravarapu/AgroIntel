import sys
import time
import os
import threading
import pickle

def kill_after(seconds):
    def _kill():
        time.sleep(seconds)
        print(f"\n[TIMEOUT] Hanging for >{seconds}s. Exiting...", file=sys.stderr)
        os._exit(1)
    threading.Thread(target=_kill, daemon=True).start()

kill_after(10)

print("Loading RandomForest.pkl with pickle...")
try:
    with open("app/models/RandomForest.pkl", "rb") as f:
        model = pickle.load(f)
    print("[SUCCESS] Loaded RandomForest.pkl with pickle")
except Exception as e:
    print(f"[ERROR] failed: {e}")

os._exit(0)
