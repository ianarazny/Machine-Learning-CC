import os
import pandas as pd

# Ruta absoluta al directorio raíz del proyecto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data"))

def convert_file(filepath):
    try:
        df = pd.read_csv(filepath, sep=";")

        for col in df.columns:
            if df[col].dtype == "object":
                if df[col].astype(str).str.contains(',', regex=False).any():
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    try:
                        df[col] = df[col].astype(float)
                    except ValueError:
                        pass

        df.to_csv(filepath, sep=";", index=False)
        print(f"[✓] Procesado: {filepath}")
    except Exception as e:
        print(f"[!] Error en {filepath}: {e}")

def traverse_and_convert(base_dir):
    print(f"Entrando a: {base_dir}")
    print("Ruta final (absoluta):", base_dir)
    print("¿Existe?:", os.path.exists(base_dir))
    print("¿Es directorio?:", os.path.isdir(base_dir))
    for root, dirs, files in os.walk(base_dir):
        print(f"Revisando en: {root}")
        for file in files:
            if file.lower().endswith((".csv", ".log.csv")):
                print(f"→ Procesando CSV: {file}")
                convert_file(os.path.join(root, file))

if __name__ == "__main__":
    traverse_and_convert(BASE_DIR)
