"""
app.py - Aplicación Streamlit para el proyecto IA-CANCER-LOGISTIC-RF.

En esta app cargaremos los modelos entrenados y permitiremos
ingresar las características de un tumor para obtener una predicción.
"""

import streamlit as st
import numpy as np

# Diccionario para mostrar nombres amigables al usuario
FRIENDLY_NAMES = {
    "radius_mean": "Radio promedio",
    "texture_mean": "Textura promedio",
    "perimeter_mean": "Perímetro promedio",
    "area_mean": "Área promedio",
    "smoothness_mean": "Suavidad promedio",
    "compactness_mean": "Compacidad promedio",
    "concavity_mean": "Concavidad promedio",
    "concave points_mean": "Puntos cóncavos promedio",
    "symmetry_mean": "Simetría promedio",
    "fractal_dimension_mean": "Dimensión fractal promedio",

    "radius_se": "Radio error estándar",
    "texture_se": "Textura error estándar",
    "perimeter_se": "Perímetro error estándar",
    "area_se": "Área error estándar",
    "smoothness_se": "Suavidad error estándar",
    "compactness_se": "Compacidad error estándar",
    "concavity_se": "Concavidad error estándar",
    "concave points_se": "Puntos cóncavos error estándar",
    "symmetry_se": "Simetría error estándar",
    "fractal_dimension_se": "Dimensión fractal error estándar",

    "radius_worst": "Radio peor caso",
    "texture_worst": "Textura peor caso",
    "perimeter_worst": "Perímetro peor caso",
    "area_worst": "Área peor caso",
    "smoothness_worst": "Suavidad peor caso",
    "compactness_worst": "Compacidad peor caso",
    "concavity_worst": "Concavidad peor caso",
    "concave points_worst": "Puntos cóncavos peor caso",
    "symmetry_worst": "Simetría peor caso",
    "fractal_dimension_worst": "Dimensión fractal peor caso",
}

RANGES = {
    "radius_mean": (6.98, 28.11),
    "texture_mean": (9.71, 39.28),
    "perimeter_mean": (43.79, 188.50),
    "area_mean": (143.5, 2501.0),

    "smoothness_mean": (0.05, 0.16),
    "compactness_mean": (0.02, 0.35),
    "concavity_mean": (0.00, 0.43),
    "concave points_mean": (0.00, 0.20),
    "symmetry_mean": (0.11, 0.30),
    "fractal_dimension_mean": (0.05, 0.10),

    "radius_se": (0.11, 4.89),
    "texture_se": (0.36, 4.88),
    "perimeter_se": (0.76, 21.98),
    "area_se": (6.8, 542.2),

    "smoothness_se": (0.000, 0.031),
    "compactness_se": (0.002, 0.135),
    "concavity_se": (0.000, 0.396),
    "concave points_se": (0.000, 0.053),
    "symmetry_se": (0.008, 0.079),
    "fractal_dimension_se": (0.002, 0.030),

    "radius_worst": (7.93, 36.04),
    "texture_worst": (12.02, 49.54),
    "perimeter_worst": (50.41, 251.20),
    "area_worst": (185.2, 4254.0),

    "smoothness_worst": (0.07, 0.22),
    "compactness_worst": (0.03, 1.06),
    "concavity_worst": (0.00, 1.25),
    "concave points_worst": (0.00, 0.29),
    "symmetry_worst": (0.16, 0.66),
    "fractal_dimension_worst": (0.06, 0.21),
}



# Importar utilidades desde el módulo de predicción
from src.predict import (
    load_models,
    FEATURE_COLUMNS,
    predict_patient,
)

# =========================================================
# Cargar modelos SOLO una vez (uso de cache de Streamlit)
# =========================================================
@st.cache_resource
def get_models():
    return load_models()

models = get_models()

# =========================================================
# Interfaz principal de la app
# =========================================================

st.set_page_config(page_title="IA-CANCER-LOGISTIC-RF", layout="centered")

st.title("Predicción de Cáncer de Mama")
st.write(
    """
    Esta aplicación permite ingresar las características de un tumor
    para obtener una predicción usando modelos de **Regresión Logística**
    y **Random Forest** entrenados con el dataset *Breast Cancer Wisconsin Diagnostic*.
    """
)

# Selector de modelo
model_option = st.selectbox(
    "Seleccione el modelo de clasificación:",
    options=["Regresión Logística (logreg)", "Random Forest (rf)"],
)

# Traducir la opción elegida al nombre interno que usa predict_patient
if "logreg" in model_option:
    selected_model_name = "logreg"
else:
    selected_model_name = "rf"

st.write(f"Modelo seleccionado internamente: `{selected_model_name}`")

# =========================================================
# Formulario de entrada para las características del tumor
# =========================================================

st.subheader("Ingrese las características del tumor")

st.caption(
    """
    Las características siguientes corresponden a mediciones obtenidas a partir 
    de imágenes digitalizadas de células mamarias. Incluyen medidas del radio, 
    textura, perímetro, área, suavidad, compacidad, concavidad, simetría y 
    dimensión fractal del tejido. Cada una se calcula como promedio, error 
    estándar o peor caso según la morfología observada.
    """
)


input_values = {}

for feature in FEATURE_COLUMNS:
    friendly = FRIENDLY_NAMES.get(feature, feature)
    min_val, max_val = RANGES.get(feature, (0.0, 100.0))

    label = f"{friendly} (rango: {min_val} – {max_val})"

    # Caja de texto para escribir el valor
    raw_value = st.text_input(
        label=label,
        value=str(min_val),
    )

    # Intentar convertir a número (aceptando coma o punto)
    try:
        raw_value_clean = raw_value.replace(",", ".")
        value = float(raw_value_clean)
        # Forzar que quede dentro del rango
        if value < min_val:
            value = min_val
        if value > max_val:
            value = max_val
    except ValueError:
        # Si el usuario escribe algo no numérico, usamos el mínimo
        value = float(min_val)

    input_values[feature] = value


# =========================================================
# Botón para realizar la predicción
# =========================================================

st.subheader("Predicción")

if st.button("Calcular predicción"):
    result = predict_patient(
        models=models,
        input_data=input_values,
        model_name=selected_model_name,
    )

    # Extraer predicción y probabilidad
    prediction = result["prediction"]
    probability = result["probability"]

    # Si vienen como arrays, tomar el primer elemento
    if isinstance(prediction, (np.ndarray, list)):
        prediction = prediction[0]
    if isinstance(probability, (np.ndarray, list)):
        probability = probability[0]

    # Traducir 0/1 a texto
    if prediction == 1:
        diagnosis_text = "Maligno"
    else:
        diagnosis_text = "Benigno"

    # Mostrar resultados al usuario
    st.markdown(f"### Diagnóstico estimado: **{diagnosis_text}**")
    st.markdown(f"**Probabilidad de malignidad:** {probability:.2%}")

    # Nota aclaratoria
    st.info(
        "⚠️ Esta herramienta tiene fines exclusivamente educativos y no debe "
        "utilizarse como sistema de diagnóstico médico real."
    )
