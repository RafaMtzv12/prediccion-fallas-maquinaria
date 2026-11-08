import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('data/ai4i2020.csv')

columnas = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear', 'Machine failure']

correlacion = df[columnas].corr()
print(correlacion)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(correlacion, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(columnas)))
ax.set_yticks(range(len(columnas)))
ax.set_xticklabels(columnas, rotation=45, ha='right')
ax.set_yticklabels(columnas)

for i in range(len(columnas)):
    for j in range(len(columnas)):
        ax.text(j, i, round(correlacion.iloc[i, j], 2), ha='center', va='center')

plt.colorbar(im)
plt.tight_layout()
plt.savefig('paso4_correlacion.png')
print("\nGráfica guardada como paso4_correlacion.png")
