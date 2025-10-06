import joblib
import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline

# Get the script directory and construct path to model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "..", "models", "regressor_flow_packets.joblib")

# Cargar el artefacto (pipeline o modelo)
art = joblib.load(model_path)

if isinstance(art, dict) and "pipeline" in art:
    pipe = art["pipeline"]
    X_cols = art.get("x_cols")  # columnas crudas usadas para entrenar
else:
    pipe = art
    # Si guardaste el pipeline directo, podés probar:
    X_cols = getattr(pipe, "feature_names_out_", None)
    if X_cols is None:
        # Si no tenés feature_names_out_, vas a necesitar pasarle X de validación
        # con las mismas columnas para poder reconstruir nombres.
        pass

# === 2) Extraer modelo final ===
est = pipe.steps[-1][1] if isinstance(pipe, Pipeline) else pipe
print("Final estimator:", type(est).__name__)

if not hasattr(est, "feature_importances_"):
    raise RuntimeError("El estimador final no expone feature_importances_. Revisa qué guardaste.")

# === 3) Reconstruir nombres de features tras el preprocesado ===
# El preprocesador está en el paso "pre" según tu notebook
pre = pipe.named_steps.get("pre")
if pre is None:
    raise RuntimeError("No se encontró el paso 'pre' en el pipeline.")

# Si conocés las columnas crudas (X_cols), mejor:
if X_cols is None:
    # Último recurso: intentar leerlas del preprocesador
    try:
        raw_names = pre.feature_names_in_
        X_cols = list(raw_names)
    except Exception:
        raise RuntimeError("No pude deducir X_cols. Pasá tus columnas crudas (las del entrenamiento).")

# Esto devuelve los nombres expandidos (num + one-hot cat)
try:
    feat_names = pre.get_feature_names_out()
except TypeError:
    # Compatibilidad con versiones viejas de sklearn/OneHotEncoder
    # En versiones viejas, get_feature_names_out() puede requerir input_features
    feat_names = pre.get_feature_names_out(input_features=X_cols)

feat_names = np.array(feat_names, dtype=str)

# === 4) Importancias del RF mapeadas a nombres ===
importances = est.feature_importances_
if len(importances) != len(feat_names):
    raise RuntimeError(
        f"Dimensión no coincide: importances={len(importances)} vs features={len(feat_names)}. "
        "Asegurate de usar las mismas columnas y versión de sklearn que al entrenar."
    )

order = np.argsort(importances)[::-1]
rank_df = pd.DataFrame({
    "feature": feat_names[order],
    "importance": importances[order]
})
print(rank_df.head(25))

rank_df.to_csv("feature_importance_random_forest.csv", index=False)
print("Guardado: feature_importance_random_forest.csv")