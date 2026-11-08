import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

# 1. Cargar datos
df = pd.read_csv('data/ai4i2020.csv')

# 2. Quitar columnas que delatan la falla (evitar fuga de datos)
df = df.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

# 3. Convertir Type (L/M/H) a columnas numericas
df = pd.get_dummies(df, columns=['Type'], dtype=int)

# 4. Separar entradas (X) y lo que se quiere predecir (y)
X = df.drop(columns=['Machine failure'])
y = df['Machine failure']

# 5. Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Entrenamiento:", X_train.shape, "| Prueba:", X_test.shape)
print("Fallas en prueba:", y_test.sum(), "de", len(y_test))

# 6. Definir los 3 modelos
modelos = {
    "Regresion Logistica": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(class_weight='balanced', random_state=42)),
}

# 7. Entrenar y evaluar
filas = []
fig, ejes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (nombre, modelo) in zip(ejes, modelos.items()):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    filas.append([nombre, acc, prec, rec, f1])

    print("\n===", nombre, "===")
    print("Matriz de confusion [[VN, FP], [FN, VP]]:")
    print(confusion_matrix(y_test, pred))

    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
    ax.set_title(nombre)

plt.tight_layout()
plt.savefig('paso5_matrices_confusion.png')
print("\nGrafica guardada como paso5_matrices_confusion.png")

# 8. Tabla comparativa
tabla = pd.DataFrame(filas, columns=['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1']).round(3)
print("\n=== COMPARACION ===")
print(tabla.to_string(index=False))

# 9. Variables mas importantes segun Random Forest
rf = modelos["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).round(3)
print("\n=== Importancia de variables (Random Forest) ===")
print(imp)

# 10. Guardar resultados en texto
with open('resultados_paso5_modelos.txt', 'w') as f:
    f.write(tabla.to_string(index=False))
    f.write("\n\nImportancia de variables (Random Forest):\n")
    f.write(imp.to_string())
print("\nResultados guardados en resultados_paso5_modelos.txt")
