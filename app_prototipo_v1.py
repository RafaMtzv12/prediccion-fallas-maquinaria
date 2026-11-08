import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predicción de fallas", page_icon="🛠️")

# 1. Cargar el modelo guardado (solo una vez)
@st.cache_resource
def cargar():
    return joblib.load('modelo_final.joblib')

paquete = cargar()
modelo = paquete['modelo']
columnas = paquete['columnas']
umbral = paquete['umbral']

# 2. Encabezado
st.title("Predicción de fallas en maquinaria industrial")
st.write("Prototipo de mantenimiento predictivo. Captura los valores de operación "
         "de la máquina y el modelo (Random Forest) estimará la probabilidad de falla.")

# 3. Formulario de captura
col1, col2 = st.columns(2)
with col1:
    tipo = st.selectbox("Tipo de producto (calidad)", ["L", "M", "H"])
    aire = st.number_input("Temperatura del aire (K)", 295.0, 305.0, 300.0, 0.1)
    proceso = st.number_input("Temperatura del proceso (K)", 305.0, 314.0, 310.0, 0.1)
with col2:
    vel = st.number_input("Velocidad de rotación (rpm)", 1000, 3000, 1500, 10)
    torque = st.number_input("Torque (Nm)", 0.0, 80.0, 40.0, 0.1)
    desgaste = st.number_input("Desgaste de herramienta (min)", 0, 260, 100, 1)

# 4. Predicción
if st.button("Predecir"):
    fila = pd.DataFrame([{
        'Air temperature': aire,
        'Process temperature': proceso,
        'Rotational speed': vel,
        'Torque': torque,
        'Tool wear': desgaste,
        'Type_H': int(tipo == 'H'),
        'Type_L': int(tipo == 'L'),
        'Type_M': int(tipo == 'M'),
    }])
    # Mismas variables derivadas que en el entrenamiento
    fila['temp_diff'] = fila['Process temperature'] - fila['Air temperature']
    fila['power'] = fila['Torque'] * fila['Rotational speed'] * 2 * np.pi / 60
    fila['wear_torque'] = fila['Tool wear'] * fila['Torque']
    fila = fila[columnas]  # mismo orden que al entrenar

    prob = modelo.predict_proba(fila)[0, 1]
    st.metric("Probabilidad de falla", f"{prob * 100:.1f}%")
    st.progress(float(prob))
    if prob >= umbral:
        st.error("⚠️ ALERTA: alta probabilidad de falla. Se recomienda revisar la máquina.")
    else:
        st.success("✅ Operación normal: baja probabilidad de falla.")
    st.caption(f"Umbral de decisión del modelo: {umbral * 100:.0f}%")
