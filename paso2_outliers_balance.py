import pandas as pd
import matplotlib
matplotlib.use('Agg')  # para guardar gráficas sin necesitar pantalla
import matplotlib.pyplot as plt

df = pd.read_csv('data/ai4i2020.csv')

# 1. Balance de clases: ¿cuántas fallas vs. no fallas?
print("Distribución de Machine failure:")
print(df['Machine failure'].value_counts())
print("\nPorcentaje de fallas:", round(df['Machine failure'].mean() * 100, 2), "%")

# 2. Boxplots para detectar outliers en variables numéricas
columnas_numericas = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, col in enumerate(columnas_numericas):
    df.boxplot(column=col, ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.savefig('paso2_boxplots_outliers.png')
print("\nGráfica guardada como paso2_boxplots_outliers.png")
