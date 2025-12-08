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

# Métricas obtenidas en el conjunto de prueba (Fase 3)
MODEL_METRICS = {
    "logreg": {
        "accuracy": 0.97,
        "precision": 0.96,
        "recall": 0.98,
        "f1_score": 0.97,
    },
    "rf": {
        "accuracy": 0.96,
        "precision": 0.95,
        "recall": 0.97,
        "f1_score": 0.96,
    },
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
    # --- Modelo 1: Regresión Logística ---
    result_logreg = predict_patient(
        models=models,
        input_data=input_values,
        model_name="logreg",
    )

    pred_logreg = result_logreg["prediction"]
    prob_logreg = result_logreg["probability"]

    # Si vienen como arrays, tomar el primer elemento
    if isinstance(pred_logreg, (np.ndarray, list)):
        pred_logreg = pred_logreg[0]
    if isinstance(prob_logreg, (np.ndarray, list)):
        prob_logreg = prob_logreg[0]

    diag_logreg = "Maligno" if pred_logreg == 1 else "Benigno"

    # --- Modelo 2: Random Forest ---
    result_rf = predict_patient(
        models=models,
        input_data=input_values,
        model_name="rf",
    )

    pred_rf = result_rf["prediction"]
    prob_rf = result_rf["probability"]

    if isinstance(pred_rf, (np.ndarray, list)):
        pred_rf = pred_rf[0]
    if isinstance(prob_rf, (np.ndarray, list)):
        prob_rf = prob_rf[0]

    diag_rf = "Maligno" if pred_rf == 1 else "Benigno"

    # Mostrar resultados en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Regresión Logística")
        st.markdown(f"**Diagnóstico:** {diag_logreg}")
        st.markdown(f"**Probabilidad de malignidad:** {prob_logreg:.2%}")

    with col2:
        st.markdown("### Random Forest")
        st.markdown(f"**Diagnóstico:** {diag_rf}")
        st.markdown(f"**Probabilidad de malignidad:** {prob_rf:.2%}")

    # Nota aclaratoria general
    st.info(
        "⚠️ Esta herramienta tiene fines exclusivamente educativos y no debe "
        "utilizarse como sistema de diagnóstico médico real."
    )

    st.markdown("---")
    st.subheader("Desempeño de los modelos en datos de prueba")

    st.caption(
        "Estas métricas se calcularon en la Fase 3 del proyecto usando el conjunto de prueba. "
        "En problemas de salud, la métrica más crítica suele ser el **recall de la clase maligna**, "
        "porque indica cuántos casos malignos reales logra detectar el modelo."
    )

    # Crear tabla simple
    metrics_table = {
        "Métrica": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Regresión Logística": [
            f"{MODEL_METRICS['logreg']['accuracy']:.2%}",
            f"{MODEL_METRICS['logreg']['precision']:.2%}",
            f"{MODEL_METRICS['logreg']['recall']:.2%}",
            f"{MODEL_METRICS['logreg']['f1_score']:.2%}",
        ],
        "Random Forest": [
            f"{MODEL_METRICS['rf']['accuracy']:.2%}",
            f"{MODEL_METRICS['rf']['precision']:.2%}",
            f"{MODEL_METRICS['rf']['recall']:.2%}",
            f"{MODEL_METRICS['rf']['f1_score']:.2%}",
        ],
    }

    st.table(metrics_table)

st.markdown("---")
st.subheader("Interpretación del desempeño de los modelos")

st.write(
    """
    Ambos modelos presentan un desempeño muy similar en términos de exactitud general 
    (**accuracy ≈ 0.9649**), lo cual indica que los dos clasifican correctamente la gran 
    mayoría de los casos del conjunto de prueba. Sin embargo, existen diferencias 
    importantes en la forma como cada modelo comete errores.

    **La Regresión Logística** ofrece un mejor equilibrio entre precisión y recall, con 
    un **F1-score ligeramente superior (0.9512)**. Su **recall (0.9286)** es más alto que 
    el de Random Forest, lo que implica que detecta un mayor número de tumores malignos, 
    reduciendo la cantidad de falsos negativos. Este comportamiento es particularmente 
    valioso en contextos médicos, donde es preferible identificar la mayor cantidad 
    posible de casos malignos.

    **Random Forest**, por otro lado, se caracteriza por tener una **precisión perfecta 
    (1.0000)**: no genera falsos positivos. Sin embargo, su **recall es menor (0.9048)**, 
    lo que indica que deja pasar algunos tumores malignos al clasificarlos como benignos. 
    Esto refleja un modelo más conservador para predecir malignidad: solo etiqueta un caso 
    como maligno cuando tiene una seguridad muy alta.

    En conjunto, ambos modelos son adecuados para el problema, pero **la Regresión Logística 
    ofrece un mejor balance general**, mientras que **Random Forest prioriza evitar falsos 
    positivos a costa de aumentar los falsos negativos**.
    """
)

