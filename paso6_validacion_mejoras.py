import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Cargar y preparar datos (igual que en el paso 5)
df = pd.read_csv('data/ai4i2020.csv')
df = df.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
df = pd.get_dummies(df, columns=['Type'], dtype=int)

y = df['Machine failure']
X_base = df.drop(columns=['Machine failure'])

# 2. Crear variables derivadas a partir de los sensores
X_extra = X_base.copy()
X_extra['temp_diff'] = X_extra['Process temperature'] - X_extra['Air temperature']
X_extra['power'] = X_extra['Torque'] * X_extra['Rotational speed'] * 2 * np.pi / 60
X_extra['wear_torque'] = X_extra['Tool wear'] * X_extra['Torque']

# 3. Los 3 modelos (el escalado va dentro del pipeline para no hacer trampa)
def crear_modelos():
    return {
        "Regresion Logistica": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(class_weight='balanced', random_state=42)),
    }

# 4. Validacion cruzada: 5 examenes con distintas partes de los datos
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metricas = {'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}

def evaluar(X, etiqueta):
    filas = []
    for nombre, modelo in crear_modelos().items():
        r = cross_validate(modelo, X, y, cv=cv, scoring=metricas)
        filas.append([etiqueta, nombre,
                      r['test_precision'].mean(), r['test_recall'].mean(),
                      r['test_f1'].mean(), r['test_f1'].std()])
    return filas

print("Evaluando con validacion cruzada (puede tardar un par de minutos)...")
filas = evaluar(X_base, "Variables originales") + evaluar(X_extra, "Con variables derivadas")
tabla = pd.DataFrame(filas, columns=['Variables', 'Modelo', 'Precision', 'Recall', 'F1', 'F1_desv']).round(3)
print("\n=== VALIDACION CRUZADA (promedio de 5 pliegues) ===")
print(tabla.to_string(index=False))

# 5. Ajuste del umbral para Random Forest con variables derivadas
rf = crear_modelos()["Random Forest"]
probs = cross_val_predict(rf, X_extra, y, cv=cv, method='predict_proba')[:, 1]

print("\n=== UMBRAL DE DECISION (Random Forest, variables derivadas) ===")
filas_u = []
for u in [0.2, 0.3, 0.4, 0.5, 0.6]:
    pred = (probs >= u).astype(int)
    filas_u.append([u,
                    round(precision_score(y, pred), 3),
                    round(recall_score(y, pred), 3),
                    round(f1_score(y, pred), 3),
                    int(((pred == 0) & (y == 1)).sum()),
                    int(((pred == 1) & (y == 0)).sum())])
tabla_u = pd.DataFrame(filas_u, columns=['Umbral', 'Precision', 'Recall', 'F1',
                                          'Fallas_perdidas', 'Falsas_alarmas'])
print(tabla_u.to_string(index=False))

# 6. Guardar resultados
with open('resultados_paso6_validacion.txt', 'w') as f:
    f.write("VALIDACION CRUZADA (5 pliegues)\n")
    f.write(tabla.to_string(index=False))
    f.write("\n\nUMBRAL DE DECISION (Random Forest, variables derivadas)\n")
    f.write(tabla_u.to_string(index=False))
print("\nResultados guardados en resultados_paso6_validacion.txt")
