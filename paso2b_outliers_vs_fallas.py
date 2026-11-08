import pandas as pd

df = pd.read_csv('data/ai4i2020.csv')

# Definir outliers usando el método del rango intercuartílico (IQR)
def marcar_outliers(columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return (df[columna] < limite_inferior) | (df[columna] > limite_superior)

df['outlier_velocidad'] = marcar_outliers('Rotational speed')
df['outlier_torque'] = marcar_outliers('Torque')

print("=== Rotational speed ===")
print("Total de outliers:", df['outlier_velocidad'].sum())
print("Tasa de falla en outliers:", round(df[df['outlier_velocidad']]['Machine failure'].mean()*100, 2), "%")
print("Tasa de falla en valores normales:", round(df[~df['outlier_velocidad']]['Machine failure'].mean()*100, 2), "%")

print("\n=== Torque ===")
print("Total de outliers:", df['outlier_torque'].sum())
print("Tasa de falla en outliers:", round(df[df['outlier_torque']]['Machine failure'].mean()*100, 2), "%")
print("Tasa de falla en valores normales:", round(df[~df['outlier_torque']]['Machine failure'].mean()*100, 2), "%")
