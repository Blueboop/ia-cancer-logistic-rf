"""
predict.py - Pipeline de predicción para el proyecto IA-CANCER-LOGISTIC-RF.

Este módulo se encarga de:
- Cargar los modelos entrenados (Regresión Logística y Random Forest).
- Cargar el scaler usado para la Regresión Logística.
- Definir funciones para hacer predicciones a partir de nuevos datos de pacientes.
"""

# ==============================
#print("El archivo predict.py se cargó correctamente") 
# ==============================

# Imports necesarios para cargar modelos y manejar datos
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict


# Ruta absoluta hacia la carpeta 'models'
BASE_DIR = Path(__file__).resolve().parents[1]   # sube desde src/ al proyecto raíz
MODELS_DIR = BASE_DIR / "models"

# Lista de columnas de entrada en el mismo orden que se usaron para entrenar los modelos
FEATURE_COLUMNS = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


# ---- Función para cargar los modelos entrenados ----

def load_models():
    """
    Carga el scaler y los modelos entrenados desde la carpeta 'models'.
    Retorna un diccionario con los objetos cargados.
    """

    # rutas de los archivos .pkl
    scaler_path = MODELS_DIR / "scaler_logreg.pkl"
    logreg_path = MODELS_DIR / "model_logreg.pkl"
    rf_path = MODELS_DIR / "model_rf.pkl"

    # cargar los objetos
    scaler = joblib.load(scaler_path)
    model_logreg = joblib.load(logreg_path)
    model_rf = joblib.load(rf_path)

    print("Modelos cargados correctamente.")

    return {
        "scaler": scaler,
        "logreg": model_logreg,
        "rf": model_rf
    }

#-------------------------------------------------------------------------



def build_input_dataframe(input_data: Dict[str, float]) -> pd.DataFrame:
    """
    Construye un DataFrame de una sola fila con las características del paciente,
    respetando el orden de columnas definido en FEATURE_COLUMNS.

    Parameters
    ----------
    input_data : dict
        Diccionario donde las llaves son los nombres de las columnas
        y los valores son los datos numéricos del paciente.

    Returns
    -------
    DataFrame de pandas con una fila y las columnas en el orden correcto.
    """

    # Verificamos que no falte ninguna columna
    missing = [col for col in FEATURE_COLUMNS if col not in input_data]
    if missing:
        raise ValueError(f"Faltan columnas en input_data: {missing}")

    # Ordenamos los valores según FEATURE_COLUMNS
    ordered_values = [input_data[col] for col in FEATURE_COLUMNS]

    df = pd.DataFrame([ordered_values], columns=FEATURE_COLUMNS)
    return df

# ==============================================
# función de predicción con Regresión Logística
# ==============================================
def predict_logistic(models, input_data: dict) -> dict:
    """
    Realiza predicciones usando el modelo de Regresión Logística.

    Parameters
    ----------
    models : dict
        Diccionario que contiene 'scaler' y 'logreg'.
    input_data : dict
        Datos del paciente.

    Returns
    -------
    dict con:
        - 'prediction': 0 (benigno) o 1 (maligno)
        - 'probability': probabilidad de clase 1 (maligno)
    """

    # 1. Construir DataFrame con formato correcto
    df = build_input_dataframe(input_data)

    # 2. Escalar los datos
    scaler = models["scaler"]
    X_scaled = scaler.transform(df)

    # 3. Modelo de regresión logística
    model = models["logreg"]

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]  # probabilidad de maligno

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }
# ============================================
# función de predicción con Random Forest
# ============================================
def predict_rf(models, input_data: dict) -> dict:
    """
    Realiza predicciones usando el modelo Random Forest.

    Parameters
    ----------
    models : dict
        Diccionario que contiene 'rf'.
    input_data : dict
        Datos del paciente.

    Returns
    -------
    dict con:
        - 'prediction': 0 (benigno) o 1 (maligno)
        - 'probability': probabilidad de clase 1 (maligno)
    """

    # 1. Construir DataFrame con formato correcto
    df = build_input_dataframe(input_data)

    # 2. Modelo Random Forest (sin escalar)
    model = models["rf"]

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]  # probabilidad de maligno

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }


# ============================================
# Función general para predecir con modelo elegido
# ============================================
def predict_patient(models, input_data: dict, model_name: str = "logreg") -> dict:
    """
    Orquesta la predicción usando el modelo especificado.

    Parameters
    ----------
    models : dict
        Diccionario con 'scaler', 'logreg' y 'rf'.
    input_data : dict
        Datos del paciente.
    model_name : str
        Nombre del modelo a usar: 'logreg' o 'rf'.

    Returns
    -------
    dict con:
        - 'model': nombre del modelo usado
        - 'prediction': 0 (benigno) o 1 (maligno)
        - 'probability': probabilidad de clase 1 (maligno)
    """

    if model_name == "logreg":
        result = predict_logistic(models, input_data)
    elif model_name == "rf":
        result = predict_rf(models, input_data)
    else:
        raise ValueError("model_name debe ser 'logreg' o 'rf'")

    # agregamos el nombre del modelo al resultado
    result["model"] = model_name
    return result


# prueba-------------------------
if __name__ == "__main__":
    # Prueba rápida del pipeline (solo para desarrollo)
    models = load_models()
    example_input = {col: 1.0 for col in FEATURE_COLUMNS}

    result_logreg = predict_patient(models, example_input, model_name="logreg")
    print("Predicción (Regresión Logística):", result_logreg)

    result_rf = predict_patient(models, example_input, model_name="rf")
    print("Predicción (Random Forest):", result_rf)

