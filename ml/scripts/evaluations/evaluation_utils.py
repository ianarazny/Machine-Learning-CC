from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import validation_curve
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

def comprehensive_evaluation(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                           business_priority='balanced', plot=True):
    """
    Evaluación robusta con optimización de threshold en conjunto de validación.
    
    business_priority: 'recall' (detectar todos los eventos), 
                      'precision' (evitar falsas alarmas), 
                      'balanced' (F1)
    """
    
    # 1. OPTIMIZAR THRESHOLD EN VALIDATION SET (no en test!)
    y_val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)
    
    # Diferentes criterios de optimización
    if business_priority == 'recall':
        # Maximizar recall manteniendo precisión > 0.3
        valid_indices = precisions >= 0.3
        if valid_indices.any():
            best_idx = np.argmax(recalls[valid_indices])
            best_threshold = thresholds[valid_indices][best_idx]
        else:
            best_threshold = thresholds[np.argmax(recalls)]
    elif business_priority == 'precision':
        # Maximizar precisión manteniendo recall > 0.5
        valid_indices = recalls >= 0.5
        if valid_indices.any():
            best_idx = np.argmax(precisions[valid_indices])
            best_threshold = thresholds[valid_indices][best_idx]
        else:
            best_threshold = thresholds[np.argmax(precisions)]
    else:  # balanced
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
    
    print(f"Threshold óptimo ({business_priority}): {best_threshold:.3f}")
    
    # 2. EVALUAR EN TEST SET con threshold fijo
    y_test_probs = model.predict_proba(X_test)[:, 1]
    y_pred_adjusted = (y_test_probs >= best_threshold).astype(int)
    
    # 3. MÉTRICAS COMPLETAS
    print("\n=== EVALUACIÓN EN TEST SET ===")
    print(classification_report(y_test, y_pred_adjusted))
    
    # Métricas adicionales importantes
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
    
    bal_acc = balanced_accuracy_score(y_test, y_pred_adjusted)
    mcc = matthews_corrcoef(y_test, y_pred_adjusted)
    
    print(f"\nMétricas adicionales:")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"Matthews Correlation Coefficient: {mcc:.3f}")
    
    # 4. ANÁLISIS DE DISTRIBUCIÓN DE PROBABILIDADES
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Matriz de confusión
        sns.heatmap(confusion_matrix(y_test, y_pred_adjusted), 
                   annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title("Matriz de Confusión")
        axes[0,0].set_xlabel("Predicción")
        axes[0,0].set_ylabel("Real")
        
        # Distribución de probabilidades
        axes[0,1].hist(y_test_probs[y_test==0], alpha=0.7, label='No Congestión', bins=30)
        axes[0,1].hist(y_test_probs[y_test==1], alpha=0.7, label='Congestión', bins=30)
        axes[0,1].axvline(best_threshold, color='red', linestyle='--', label='Threshold')
        axes[0,1].set_xlabel("Probabilidad predicha")
        axes[0,1].set_ylabel("Frecuencia")
        axes[0,1].set_title("Distribución de Probabilidades")
        axes[0,1].legend()
        
        # Curva Precision-Recall
        test_precisions, test_recalls, _ = precision_recall_curve(y_test, y_test_probs)
        axes[1,0].plot(test_recalls, test_precisions, marker='.')
        axes[1,0].set_xlabel("Recall")
        axes[1,0].set_ylabel("Precision")
        axes[1,0].set_title("Curva Precision-Recall (Test)")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_test_probs)
        roc_auc = auc(fpr, tpr)
        axes[1,1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[1,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1,1].set_xlabel("Tasa de Falsos Positivos")
        axes[1,1].set_ylabel("Tasa de Verdaderos Positivos")
        axes[1,1].set_title("Curva ROC")
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    # 5. ANÁLISIS DE ESTABILIDAD DEL THRESHOLD
    print(f"\n=== ANÁLISIS DE SENSIBILIDAD ===")
    for offset in [-0.05, 0, 0.05]:
        test_threshold = np.clip(best_threshold + offset, 0, 1)
        test_preds = (y_test_probs >= test_threshold).astype(int)
        f1 = 2 * np.sum((test_preds == 1) & (y_test == 1)) / (np.sum(test_preds) + np.sum(y_test) + 1e-8)
        print(f"Threshold {test_threshold:.3f}: F1 = {f1:.3f}")
    
    return y_pred_adjusted, best_threshold, {
        'balanced_accuracy': bal_acc,
        'mcc': mcc,
        'probabilities': y_test_probs
    }