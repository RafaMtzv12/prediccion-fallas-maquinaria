# Predicción de fallas en maquinaria industrial con Machine Learning

Proyecto de investigación de la Facultad de Sistemas, Universidad Autónoma de Coahuila.
Autor: Rafael de Jesús Martínez Velez. Asesor: Ernesto Liñan García.

Sistema que estima la probabilidad de falla de una máquina a partir de sus sensores
(temperaturas, velocidad de rotación, torque y desgaste de herramienta). Incluye el
análisis de datos, la comparación de tres modelos y un prototipo de monitoreo con
máquinas simuladas.

## Resultados del modelo

Se compararon Regresión Logística, Random Forest y SVM con el conjunto de datos AI4I 2020
(10,000 registros, 3.39% de fallas). El mejor fue Random Forest con tres variables derivadas
(diferencia de temperatura, potencia y desgaste por torque).

| Evaluación | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Validación cruzada (5 pliegues) | 0.895 | 0.832 | 0.861 |
| Prueba final (2,000 registros no vistos) | 0.948 | 0.809 | 0.873 |

## Estructura

- `data/ai4i2020.csv`: conjunto de datos.
- `limpieza_eda.py` y `paso2` a `paso4`: calidad de datos y análisis exploratorio.
- `paso5_entrenamiento.py`: primera comparación de los tres modelos.
- `paso6_validacion_mejoras.py`: variables derivadas, validación cruzada y umbral.
- `paso7_modelo_final.py`: entrena y guarda el modelo final (`modelo_final.joblib`).
- `paso8_pruebas_prototipo.py`: prueba con 10 registros reales no vistos.
- `simulador.py`: simula cinco máquinas que envían lecturas a una base de datos SQLite.
- `app_monitoreo.py`: aplicación de monitoreo (Streamlit).
- `app_prototipo_v1.py`: primera versión, un formulario de predicción individual.

## Cómo ejecutarlo

Requiere Python 3.12.

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

El modelo guardado se creó con las versiones de `requirements.txt`; usar otra versión de
scikit-learn puede dar errores al cargarlo. En ese caso, se regenera con
`python paso7_modelo_final.py`.

Para el monitoreo se usan dos terminales. En la primera:

    python simulador.py

En la segunda, unos segundos después:

    streamlit run app_monitoreo.py

Luego se abre http://localhost:8501 en el navegador.

## Aclaraciones

- Las lecturas en vivo son simuladas. El comportamiento de las máquinas, como la degradación
  de la Fresadora 2, se definió al programar el simulador. Por eso demuestran el sistema, pero
  no miden el desempeño del modelo, que se evaluó con los datos reales de AI4I 2020.
- AI4I 2020 es un conjunto de datos sintético. Con maquinaria real el desempeño podría ser menor.
- El modelo califica cada lectura por separado; no pronostica fallas futuras.

## Datos

Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository.
https://doi.org/10.24432/C5HS5C
