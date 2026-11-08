from ucimlrepo import fetch_ucirepo
import pandas as pd

# Cargar el dataset
ds = fetch_ucirepo(id=601)
X = ds.data.features
y = ds.data.targets

# Unir todo en una sola tabla
df = pd.concat([X, y], axis=1)

# Guardarlo en un CSV local para no tener que descargarlo cada vez
df.to_csv('data/ai4i2020.csv', index=False)

print("Tamaño del dataset:", df.shape)
print("\nTipos de datos:")
print(df.dtypes)
print("\n¿Hay valores nulos/vacíos?")
print(df.isnull().sum())
print("\n¿Hay filas duplicadas?")
print(df.duplicated().sum())
print("\nEstadísticas básicas:")
print(df.describe())
