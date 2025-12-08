"""
app.py - Aplicación Streamlit para el proyecto IA-CANCER-LOGISTIC-RF.

En esta app cargaremos los modelos entrenados y permitiremos
ingresar las características de un tumor para obtener una predicción.
"""

import streamlit as st

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
