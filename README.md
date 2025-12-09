# IA-CANCER-LOGISTIC-RF

Proyecto de clasificación supervisada que predice si un tumor de mama es **benigno** o **maligno** usando modelos de **Regresión Logística** y **Random Forest** a partir del dataset *Breast Cancer Wisconsin Diagnostic*.

##  Objetivo del proyecto

Desarrollar una herramienta de apoyo educativo que, a partir de características morfológicas de un tumor obtenidas de imágenes de tejido mamario, permita:

- Entrenar y comparar dos modelos de clasificación supervisada: **Regresión Logística** y **Random Forest**.
- Evaluar su desempeño con métricas adecuadas para problemas de salud (accuracy, precision, recall, F1-score).
- Integrar el mejor modelo (y su comparativo) en una aplicación web sencilla desarrollada con **Streamlit**, que reciba los datos de un paciente y entregue una predicción (benigno/maligno) junto con la probabilidad asociada.

## 📚 Descripción de la base de datos

Para el desarrollo del proyecto utilicé un conjunto de datos público denominado **Breast Cancer Wisconsin (Diagnostic) Dataset**, disponible en la plataforma Kaggle. La base proviene del repositorio educativo del autor:

🔗 **https://www.kaggle.com/datasets/erdemtaha/cancer-data**

Este conjunto de datos contiene información clínica y características morfológicas obtenidas de imágenes digitales de tejido mamario. Cada registro corresponde a una paciente e incluye:

- **id:** Identificador único de cada paciente.  
- **diagnosis:** Tipo de cáncer diagnosticado:  
  - **M**: *Malignant* (maligno)  
  - **B**: *Benign* (benigno)

Además, incorpora **30 variables numéricas** que describen propiedades de la masa tumoral, calculadas mediante análisis digital de imágenes, tales como:

- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`  
- `smoothness_mean`, `compactness_mean`, `concavity_mean`,  
- `concave points_mean`, entre otras características relacionadas con forma, textura e irregularidad del tumor.

Estas medidas son ampliamente utilizadas en el ámbito clínico y computacional para desarrollar modelos de clasificación orientados al **diagnóstico asistido de cáncer de mama**.

El dataset fue elaborado con fines educativos y de investigación, y corresponde a una copia del conjunto original publicado por el **UCI Machine Learning Repository**, uno de los repositorios más utilizados en estudios de Machine Learning. La licencia del conjunto es **CC BY-NC-SA 4.0**, lo que permite su uso académico y no comercial.

###  Resumen estructural del dataset

- **Total de registros:** 569 pacientes  
- **Columnas originales:** 33  
- **Columnas utilizadas en el modelado:** 30 características numéricas  
- **Variable objetivo:** `diagnosis`  
  - **B → 0** (Benigno)  
  - **M → 1** (Maligno)  
- **Tipo de problema:** Clasificación binaria  
- **Tamaño final de cada conjunto tras la división train/test (80/20):**  
  - `X_train`: 455 muestras  
  - `X_test`: 114 muestras

##  Estructura del proyecto

El repositorio sigue una organización modular que facilita la lectura, el mantenimiento y la reproducción de los resultados.

```
IA-CANCER-LOGISTIC-RF/
│
├─ data/ # Dataset utilizado para el entrenamiento
│ └─ Cancer_Data.csv
│
├─ notebooks/ # Desarrollo del análisis y entrenamiento
│ ├─ 01_eda.ipynb # Exploración y limpieza inicial
│ ├─ 02_preprocess.ipynb # Preprocesamiento y división train/test
│ └─ 03_modelos.ipynb # Entrenamiento y evaluación de modelos
│
├─ src/ # Código fuente reutilizable
│ └─ predict.py # Pipeline de predicción para la app
│
├─ models/ # Artefactos generados en el entrenamiento
│ ├─ scaler_logreg.pkl # Escalador usado en Regresión Logística
│ ├─ model_logreg.pkl # Modelo de Regresión Logística entrenado
│ └─ model_rf.pkl # Modelo Random Forest entrenado
│
├─ app.py # Aplicación Streamlit para predicción interactiva
├─ requirements.txt # Dependencias necesarias para ejecutar la app
└─ README.md # Documentación principal del proyecto
```
##  Fases del proyecto

El desarrollo del proyecto se realizó siguiendo cinco fases consecutivas, que garantizan un flujo de trabajo reproducible y coherente con buenas prácticas de Machine Learning.

### **🔹 Fase 1 – Exploración y análisis inicial (EDA)**
- Revisión de la estructura del dataset.
- Eliminación de columnas irrelevantes (`id`, `Unnamed: 32`).
- Análisis de la variable objetivo y detección de desbalance.
- Identificación de outliers naturales en datos biomédicos.
- Estudio de correlaciones entre variables y detección de multicolinealidad.

### **🔹 Fase 2 – Preprocesamiento**
- Codificación de la variable objetivo (B → 0, M → 1).
- Separación en variables predictoras y objetivo.
- División en conjuntos de entrenamiento y prueba (80/20).
- Aplicación de `StandardScaler` exclusivamente para Regresión Logística.
- Guardado del escalador entrenado para uso posterior.

### **🔹 Fase 3 – Entrenamiento y comparación de modelos**
Modelos entrenados:
- **Regresión Logística** (con datos escalados).
- **Random Forest** (con datos sin escalar).

Se evaluaron métricas como accuracy, precision, recall y F1-score, junto con matrices de confusión.  
Hallazgos clave:
- La Regresión Logística obtuvo mejor **recall**, útil para detectar casos malignos.
- Random Forest logró **precisión perfecta**, evitando falsos positivos.

### **🔹 Fase 4 – Construcción del pipeline de predicción**
- Implementación del módulo `src/predict.py`.
- Carga estructurada de modelos entrenados y del escalador.
- Construcción de DataFrames ordenados según `FEATURE_COLUMNS`.
- Implementación de funciones de predicción para cada modelo.
- Definición de una función unificada (`predict_patient`) lista para integrarse con una app.

### **🔹 Fase 5 – Desarrollo de la aplicación (Streamlit)**
- Creación del archivo `app.py`.
- Generación de una interfaz intuitiva para ingresar las 30 características del tumor.
- Validación automática de rangos para evitar entradas inválidas.
- Predicción simultánea con ambos modelos (LogReg y Random Forest).
- Visualización clara del diagnóstico y probabilidad de malignidad.
- Inclusión de advertencia de uso educativo.

##  Resultados de los modelos

A continuación se presentan las métricas obtenidas por los dos modelos entrenados: **Regresión Logística** y **Random Forest**. Las métricas se evaluaron sobre el conjunto de prueba (20% del dataset).

###  Métricas de desempeño

| Modelo               | Accuracy | Precisión | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Regresión Logística | 0.9649   | 0.9750    | 0.9286 | 0.9512   |
| Random Forest       | 0.9649   | 1.0000    | 0.9048 | 0.9500   |

###  Interpretación de los resultados

- Ambos modelos alcanzan un desempeño muy similar en cuanto a **accuracy**.
- **Regresión Logística** presenta mejor **recall**, lo que indica que detecta un mayor número de tumores malignos.  
   Esto es relevante en contextos médicos donde los **falsos negativos** son críticos.
- **Random Forest** obtiene una **precisión perfecta**, es decir, no clasifica tumores benignos como malignos.  
   Útil en escenarios donde se quiere minimizar falsos positivos.
- La comparación confirma que no existe un “modelo ganador absoluto”: cada uno es fuerte en una dimensión distinta.

###  Conclusión técnica
Para fines del proyecto y de la aplicación desarrollada, se decidió mostrar **ambas predicciones en paralelo** dentro de la app, permitiendo al usuario observar:

- el diagnóstico de cada modelo,  
- y la probabilidad de malignidad correspondiente.

##  Cómo ejecutar el proyecto

A continuación se describen los pasos necesarios para instalar las dependencias y ejecutar la aplicación de predicción desarrollada con Streamlit.

---

###  Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/IA-CANCER-LOGISTIC-RF.git
cd IA-CANCER-LOGISTIC-RF
```

### Instalar dependencias
```bash
pip install -r requirements.txt
```

### Ejecutar la aplicación: 

```bash
streamlit run app.py
```

## Autoría

**Proyecto desarrollado por:**
Alexa Guzman;
Jeans Gomez;
Kevin Pepinosa.

Como parte del curso de Inteligencia Artificial.

