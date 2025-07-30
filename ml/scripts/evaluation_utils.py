from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_with_optimal_threshold(model, X_test, y_test, plot=True):
    """
    Evalúa un modelo clasificando con un threshold que maximiza el F1-score.
    
    Parámetros:
    - model: clasificador entrenado (debe implementar predict_proba)
    - X_test: conjunto de prueba (escalado o no, según corresponda)
    - y_test: etiquetas reales del conjunto de prueba
    - plot: si True, muestra matriz de confusión

    Devuelve:
    - y_pred_adjusted: predicciones binarias ajustadas
    - best_threshold: threshold óptimo calculado
    """
    # Verificación
    if not hasattr(model, "predict_proba"):
        raise ValueError("El modelo debe implementar predict_proba")

    # Obtener probabilidades de la clase positiva
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calcular precisión, recall y thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)

    # Encontrar threshold óptimo
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"\n Threshold óptimo (max F1-score): {best_threshold:.2f}")

    # Clasificar con el nuevo threshold
    y_pred_adjusted = (y_probs > best_threshold).astype(int)

    # Reporte de clasificación
    print("\n Reporte de clasificación (threshold ajustado):")
    print(classification_report(y_test, y_pred_adjusted))

    # Matriz de confusión
    if plot:
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred_adjusted), annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicción")
        plt.ylabel("Valor real")
        plt.title("Matriz de Confusión con Threshold Ajustado")
        plt.show()

    return y_pred_adjusted, best_threshold
