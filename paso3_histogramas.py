import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('data/ai4i2020.csv')

columnas_numericas = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for i, col in enumerate(columnas_numericas):
    df[df['Machine failure'] == 0][col].hist(ax=axes[i], bins=30, alpha=0.5, label='No falló')
    df[df['Machine failure'] == 1][col].hist(ax=axes[i], bins=30, alpha=0.5, label='Falló')
    axes[i].set_title(col)
    axes[i].legend()

plt.tight_layout()
plt.savefig('paso3_histogramas_fallas.png')
print("Gráfica guardada como paso3_histogramas_fallas.png")
