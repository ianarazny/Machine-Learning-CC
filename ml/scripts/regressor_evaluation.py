import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
import json, joblib
import os
BASE_DIR = os.path.dirname(__file__)

# ---------- MÉTRICAS BÁSICAS ----------
def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100.0 * np.mean(num / (den + eps))

def ape_percent(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + eps)

def within_tolerance_accuracy(y_true, y_pred, tol=0.05):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(1.0, np.abs(y_true))
    ok = np.abs(y_pred - y_true) <= tol * denom
    return 100.0 * np.mean(ok)

# ---------- EVALUACIÓN GLOBAL ----------
def regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    ape_all   = ape_percent(y_true, y_pred)
    ape_med   = np.nanmedian(ape_all)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "SMAPE": smape_val,
        "APE_mediana": ape_med
    }

# ---------- EVALUACIÓN CON BINS ----------
def classification_like_report(y_true, y_pred, bins=None, labels=None):
    if bins is None:
        bins = [0, 1e5, 1e6, 1e7, np.inf]
    if labels is None:
        labels = [0, 1, 2, 3]
    y_true_bins = np.digitize(y_true, bins) - 1
    y_pred_bins = np.digitize(y_pred, bins) - 1
    return classification_report(y_true_bins, y_pred_bins, target_names=[str(l) for l in labels])

# ---------- CARGA DE MODELO Y EVALUACIÓN ----------

# Leer el manifest para saber qué formato se guardó
with open("data_analysis/holdout_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

artifact = joblib.load(os.path.join(BASE_DIR, "data_analysis", f"regressor_flow_{manifest['target']}.joblib"))
pipe = artifact["pipeline"]
X_cols = artifact["x_cols"]

# Cargar X_test según el formato
if manifest["saved_as"]["X_test"] == "parquet":
    X_test = pd.read_parquet(os.path.join(BASE_DIR, "data_analysis", "X_test.parquet"), engine="fastparquet")
else:
    X_test = pd.read_csv("X_test.csv")

# Cargar y_test
y_test = np.load(os.path.join(BASE_DIR, "data_analysis", "y_test.npy"))

# Asegurar orden de columnas
X_test = X_test[X_cols]
y_pred = pipe.predict(X_test)

# Métricas globales
metrics = regression_metrics(y_test, y_pred)
print("=== Métricas Globales ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:,.4f}")
    else:
        print(f"{k}: {v}")

# Exactitud dentro de tolerancias
for tol in [0.01, 0.05, 0.10]:
    acc = within_tolerance_accuracy(y_test, y_pred, tol=tol)
    print(f"Exactitud ±{int(tol*100)}%: {acc:.2f} %")

# Reporte por bins (tipo clasificación)
print("\n=== Reporte por bins ===")
print(classification_like_report(y_test, y_pred))
