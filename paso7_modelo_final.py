import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

UMBRAL = 0.5

# 1. Preparar datos (igual que en pasos anteriores)
df = pd.read_csv('data/ai4i2020.csv')
df = df.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
df = pd.get_dummies(df, columns=['Type'], dtype=int)

y = df['Machine failure']
X = df.drop(columns=['Machine failure'])
X['temp_diff'] = X['Process temperature'] - X['Air temperature']
X['power'] = X['Torque'] * X['Rotational speed'] * 2 * np.pi / 60
X['wear_torque'] = X['Tool wear'] * X['Torque']

# 2. Dividir 80% entrenamiento / 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Entrenar el modelo final
modelo = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

# 4. Evaluar con datos que nunca vio
probs = modelo.predict_proba(X_test)[:, 1]
pred = (probs >= UMBRAL).astype(int)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

print("=== MODELO FINAL: Random Forest + variables derivadas, umbral", UMBRAL, "===")
print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
print("Matriz de confusion [[VN, FP], [FN, VP]]:")
print(cm)

# 5. Importancia de variables
imp = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False).round(3)
print("\n=== Importancia de variables ===")
print(imp)

# 6. Graficas
fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ejes[0], colorbar=False)
ejes[0].set_title("Matriz de confusion (modelo final)")
imp.sort_values().plot(kind='barh', ax=ejes[1])
ejes[1].set_title("Importancia de variables")
plt.tight_layout()
plt.savefig('paso7_modelo_final.png')
print("\nGrafica guardada como paso7_modelo_final.png")

# 7. Guardar el modelo y su configuracion para el prototipo
joblib.dump({'modelo': modelo, 'columnas': list(X.columns), 'umbral': UMBRAL},
            'modelo_final.joblib')
print("Modelo guardado como modelo_final.joblib")

with open('resultados_paso7_modelo_final.txt', 'w') as f:
    f.write(f"Accuracy {acc:.3f} | Precision {prec:.3f} | Recall {rec:.3f} | F1 {f1:.3f}\n")
    f.write(f"Matriz de confusion:\n{cm}\n\nImportancia de variables:\n{imp.to_string()}\n")
