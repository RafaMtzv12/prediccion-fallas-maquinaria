import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# 1. Cargar el modelo guardado
paquete = joblib.load('modelo_final.joblib')
modelo, columnas, umbral = paquete['modelo'], paquete['columnas'], paquete['umbral']

# 2. Preparar los datos igual que en el entrenamiento
df = pd.read_csv('data/ai4i2020.csv')
df = df.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
df = pd.get_dummies(df, columns=['Type'], dtype=int)
y = df['Machine failure']
X = df.drop(columns=['Machine failure'])
X['temp_diff'] = X['Process temperature'] - X['Air temperature']
X['power'] = X['Torque'] * X['Rotational speed'] * 2 * np.pi / 60
X['wear_torque'] = X['Tool wear'] * X['Torque']

# 3. Mismo corte 80/20 de siempre: X_test son registros que el modelo nunca vio
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Elegir 5 registros con falla y 5 sin falla
fallas = X_test[y_test == 1].sample(5, random_state=1)
normales = X_test[y_test == 0].sample(5, random_state=1)
muestra = pd.concat([fallas, normales])
real = y_test.loc[muestra.index]

# 5. Predecir
probs = modelo.predict_proba(muestra[columnas])[:, 1]
pred = (probs >= umbral).astype(int)

tabla = pd.DataFrame({
    'Velocidad': muestra['Rotational speed'].values,
    'Torque': muestra['Torque'].values,
    'Desgaste': muestra['Tool wear'].values,
    'Real': real.map({0: 'Sin falla', 1: 'Falla'}).values,
    'Prob_falla_%': (probs * 100).round(1),
    'Prediccion': pd.Series(pred).map({0: 'Sin falla', 1: 'Falla'}).values,
})
tabla['Acierto'] = np.where(real.values == pred, 'SI', 'NO')

print(tabla.to_string(index=False))
print(f"\nAciertos: {(real.values == pred).sum()} de {len(tabla)}")

with open('resultados_paso8_pruebas.txt', 'w') as f:
    f.write(tabla.to_string(index=False))
    f.write(f"\n\nAciertos: {(real.values == pred).sum()} de {len(tabla)}\n")
print("Guardado en resultados_paso8_pruebas.txt")
