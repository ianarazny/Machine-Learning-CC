# scripts/preprocess_all.py
import os
import pandas as pd
from preprocessing import compute_metrics

# Configure paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_processed"))

os.makedirs(OUT_DIR, exist_ok=True)

def preprocess_all():
    print(f"🔍 Looking in: {BASE_DIR}")
    print("📂 Found files:")
    for root, _, files in os.walk(BASE_DIR):
        print(f" - {root}")
        for file in files:
            print(f"   • {file}")
            if file.endswith(".csv") or file.endswith(".log.csv"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, BASE_DIR)
                out_path = os.path.join(OUT_DIR, rel_path)

                try:
                    df = pd.read_csv(full_path, sep=";")
                    df_clean = compute_metrics(df)

                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    df_clean.to_csv(out_path, index=False)
                    print(f"[✓] Processed and saved: {out_path}")

                except Exception as e:
                    print(f"[!] Error in {full_path}: {e}")

if __name__ == "__main__":
    preprocess_all()