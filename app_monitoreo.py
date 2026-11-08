import sqlite3

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DB = 'monitoreo.db'

st.set_page_config(page_title="Monitoreo de maquinaria", page_icon="🛠️", layout="wide")


@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_final.joblib')


@st.cache_data
def referencia_normal():
    # Promedio y desviacion de las maquinas SIN falla del conjunto de datos real
    df = pd.read_csv('data/ai4i2020.csv')
    normal = df[df['Machine failure'] == 0]
    cols = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
    return normal[cols].agg(['mean', 'std']).T


paquete = cargar_modelo()
modelo = paquete['modelo']
columnas = paquete['columnas']
umbral_modelo = paquete['umbral']


def nombres_maquinas():
    try:
        con = sqlite3.connect(DB, timeout=10)
        filas = con.execute("SELECT nombre FROM maquinas ORDER BY id").fetchall()
        con.close()
        return [f[0] for f in filas]
    except sqlite3.Error:
        return []


def leer_lecturas(limite):
    try:
        con = sqlite3.connect(DB, timeout=10)
        df = pd.read_sql_query(
            "SELECT l.id, l.ts, l.maquina_id, m.nombre, m.tipo, l.air, l.proc, l.rpm, l.torque, l.wear "
            "FROM lecturas l JOIN maquinas m ON m.id = l.maquina_id "
            "ORDER BY l.id DESC LIMIT ?", con, params=(limite,))
        con.close()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df['ts'] = pd.to_datetime(df['ts'])
    return df.sort_values('id').reset_index(drop=True)


def predecir(df):
    # Mismo procesamiento que en el entrenamiento
    X = pd.DataFrame({
        'Air temperature': df['air'],
        'Process temperature': df['proc'],
        'Rotational speed': df['rpm'],
        'Torque': df['torque'],
        'Tool wear': df['wear'],
        'Type_H': (df['tipo'] == 'H').astype(int),
        'Type_L': (df['tipo'] == 'L').astype(int),
        'Type_M': (df['tipo'] == 'M').astype(int),
    })
    X['temp_diff'] = X['Process temperature'] - X['Air temperature']
    X['power'] = X['Torque'] * X['Rotational speed'] * 2 * np.pi / 60
    X['wear_torque'] = X['Tool wear'] * X['Torque']
    return modelo.predict_proba(X[columnas])[:, 1]


# ---------- Controles laterales ----------
st.sidebar.title("Controles")
maquina = st.sidebar.selectbox("Máquina", ["Todas"] + nombres_maquinas())
umbral = st.sidebar.slider(
    "Umbral de alerta", 0.10, 0.90, float(umbral_modelo), 0.05,
    help="Si la probabilidad de falla alcanza este valor, se marca alerta. "
         "Bajarlo detecta más fallas pero genera más falsas alarmas.")
ventana = st.sidebar.slider("Lecturas a mostrar por máquina", 20, 300, 60, 10)
auto = st.sidebar.toggle("Actualizar automáticamente", value=True)
seg = st.sidebar.slider("Actualizar cada (segundos)", 1, 10, 2)

st.title("Monitoreo de maquinaria industrial")
st.caption("Las lecturas en vivo provienen de un simulador de máquinas (simulador.py). "
           "El modelo fue entrenado y evaluado con datos reales del conjunto AI4I 2020.")

tabs = st.tabs(["Monitoreo en vivo", "Historial de alertas", "Rendimiento del modelo"])


# ---------- Pestaña 1: monitoreo en vivo ----------
@st.fragment(run_every=(seg if auto else None))
def panel_en_vivo():
    df = leer_lecturas(ventana * 5)
    if df.empty:
        st.info("Aún no hay lecturas. Abre otra terminal e inicia el simulador: python simulador.py")
        return
    df['prob'] = predecir(df)
    ultimas = df.groupby('maquina_id').tail(1).sort_values('maquina_id')

    cols = st.columns(len(ultimas))
    for c, (_, r) in zip(cols, ultimas.iterrows()):
        alerta = r['prob'] >= umbral
        c.metric(f"{'🔴' if alerta else '🟢'} {r['nombre']}",
                 f"{r['prob'] * 100:.0f}%",
                 "ALERTA" if alerta else "Normal",
                 delta_color="inverse" if alerta else "off")
    st.caption(f"Última lectura: {df['ts'].max():%H:%M:%S}")

    etiquetas = {'ts': 'Hora', 'prob': 'Probabilidad de falla', 'nombre': 'Máquina'}
    if maquina == "Todas":
        fig = px.line(df, x='ts', y='prob', color='nombre', labels=etiquetas,
                      title="Probabilidad de falla por máquina")
        fig.add_hline(y=umbral, line_dash="dash", annotation_text="umbral")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, key="prob_todas")
        return

    d = df[df['nombre'] == maquina]
    if d.empty:
        st.warning("Esta máquina aún no tiene lecturas.")
        return
    actual = d.iloc[-1]
    if actual['prob'] >= umbral:
        st.error(f"⚠️ ALERTA en {maquina}: alta probabilidad de falla. Se recomienda revisarla.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de falla actual", f"{actual['prob'] * 100:.1f}%")
    c2.metric("Estado", "ALERTA" if actual['prob'] >= umbral else "Normal")
    c3.metric("Alertas en esta ventana", int((d['prob'] >= umbral).sum()))

    fig = px.line(d, x='ts', y='prob', labels=etiquetas, title=f"Probabilidad de falla - {maquina}")
    fig.add_hline(y=umbral, line_dash="dash", annotation_text="umbral")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, key="prob_maquina")

    sensores = [('torque', 'Torque (Nm)'), ('rpm', 'Velocidad (rpm)'),
                ('wear', 'Desgaste de herramienta (min)'), ('proc', 'Temperatura del proceso (K)')]
    izq, der = st.columns(2)
    for i, (col, nombre_s) in enumerate(sensores):
        f = px.line(d, x='ts', y=col, labels={'ts': 'Hora', col: nombre_s}, title=nombre_s)
        (izq if i % 2 == 0 else der).plotly_chart(f, key=f"sensor_{col}")

    st.markdown("**¿Qué tan fuera de lo normal está cada sensor?** "
                "(comparado con máquinas sin falla del conjunto de datos real)")
    ref = referencia_normal()
    pares = [("Temperatura del aire (K)", 'air', 'Air temperature'),
             ("Temperatura del proceso (K)", 'proc', 'Process temperature'),
             ("Velocidad de rotación (rpm)", 'rpm', 'Rotational speed'),
             ("Torque (Nm)", 'torque', 'Torque'),
             ("Desgaste de herramienta (min)", 'wear', 'Tool wear')]
    filas = []
    for nombre_s, col_df, col_ref in pares:
        z = (actual[col_df] - ref.loc[col_ref, 'mean']) / ref.loc[col_ref, 'std']
        filas.append({"Sensor": nombre_s,
                      "Valor actual": round(float(actual[col_df]), 1),
                      "Promedio normal": round(float(ref.loc[col_ref, 'mean']), 1),
                      "Desviaciones del promedio": round(float(z), 2),
                      "Estado": "⚠️ Fuera de lo normal" if abs(z) >= 2 else "Normal"})
    st.dataframe(pd.DataFrame(filas), hide_index=True)


with tabs[0]:
    panel_en_vivo()


# ---------- Pestaña 2: historial de alertas ----------
with tabs[1]:
    st.subheader("Historial de alertas")
    st.button("Actualizar historial")
    hist = leer_lecturas(3000)
    if hist.empty:
        st.info("Aún no hay lecturas guardadas.")
    else:
        hist['prob'] = predecir(hist)
        alertas = hist[hist['prob'] >= umbral].sort_values('id', ascending=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Lecturas analizadas", len(hist))
        c2.metric("Alertas", len(alertas))
        c3.metric("% en alerta", f"{len(alertas) / len(hist) * 100:.1f}%")
        if alertas.empty:
            st.success("Sin alertas con el umbral actual.")
        else:
            conteo = alertas.groupby('nombre').size().reset_index(name='alertas')
            st.plotly_chart(px.bar(conteo, x='nombre', y='alertas',
                                   labels={'nombre': 'Máquina', 'alertas': 'Alertas'},
                                   title="Alertas por máquina"), key="alertas_maquina")
            tabla = alertas[['ts', 'nombre', 'prob', 'rpm', 'torque', 'wear']].copy()
            tabla['prob'] = (tabla['prob'] * 100).round(1)
            tabla.columns = ['Hora', 'Máquina', 'Prob. falla (%)', 'Velocidad (rpm)',
                             'Torque (Nm)', 'Desgaste (min)']
            st.dataframe(tabla, hide_index=True)
            st.download_button("Descargar alertas (CSV)", tabla.to_csv(index=False).encode('utf-8'),
                               "alertas.csv", "text/csv")


# ---------- Pestaña 3: rendimiento del modelo ----------
with tabs[2]:
    st.subheader("Rendimiento del modelo (datos reales AI4I 2020)")
    st.caption("Estas métricas vienen de los datos reales del conjunto de prueba, no del simulador.")

    st.markdown("**Comparación de los tres modelos** (validación cruzada de 5 pliegues, con variables derivadas)")
    st.dataframe(pd.DataFrame({
        'Modelo': ['Regresión Logística', 'SVM', 'Random Forest (modelo final)'],
        'Precision': [0.164, 0.293, 0.895],
        'Recall': [0.850, 0.861, 0.832],
        'F1': [0.275, 0.438, 0.861]}), hide_index=True)

    st.markdown("**Prueba final del Random Forest** (2,000 registros que no vio al entrenar, 68 fallas)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "0.992")
    m2.metric("Precision", "0.948")
    m3.metric("Recall", "0.809")
    m4.metric("F1-score", "0.873")
    st.dataframe(pd.DataFrame([[1929, 3], [13, 55]],
                              index=['Real sin falla', 'Real con falla'],
                              columns=['Predicho sin falla', 'Predicho con falla']))

    st.markdown("**Efecto del umbral** (Random Forest, validación cruzada, 339 fallas reales)")
    st.dataframe(pd.DataFrame({
        'Umbral': [0.2, 0.3, 0.4, 0.5, 0.6],
        'Precision': [0.542, 0.700, 0.808, 0.892, 0.941],
        'Recall': [0.876, 0.853, 0.847, 0.832, 0.802],
        'F1': [0.670, 0.769, 0.827, 0.861, 0.866],
        'Fallas no detectadas': [42, 50, 52, 57, 67],
        'Falsas alarmas': [251, 124, 68, 34, 17]}), hide_index=True)

    st.image('paso7_modelo_final.png', caption="Matriz de confusión e importancia de variables del modelo final")
    st.image('paso5_matrices_confusion.png', caption="Matrices de confusión de los tres modelos")
