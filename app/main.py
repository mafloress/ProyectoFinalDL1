import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# --- Configuración de la Página ---
st.set_page_config(page_title="Predicción de Suscripción Bancaria", layout="wide")

# --- Definición de Columnas (CRUCIAL) ---
# Estas deben ser TODAS las columnas que el modelo espera, en el ORDEN EXACTO.
# Incluye las características numéricas originales (que serán escaladas) y
# todas las columnas resultantes del One-Hot Encoding de las categóricas.

# Columnas numéricas originales que fueron escaladas
NUMERICAL_FEATURES_SCALED = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Columnas categóricas originales
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Lista completa de columnas esperadas por el modelo después de One-Hot Encoding y escalado
# El orden es: numéricas escaladas primero, luego las one-hot-encoded en el orden de CATEGORICAL_FEATURES
# y dentro de cada una, el orden de sus posibles valores.
# Esto debe coincidir EXACTAMENTE con el DataFrame usado en el entrenamiento.

# Valores únicos para cada característica categórica (basado en bank-full.csv y EDA)
JOB_VALS = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
MARITAL_VALS = ['divorced', 'married', 'single'] # 'unknown' no está en marital según UCI, si estuviera se añade
EDUCATION_VALS = ['primary', 'secondary', 'tertiary', 'unknown'] # Simplificado
DEFAULT_VALS = ['no', 'yes']
HOUSING_VALS = ['no', 'yes']
LOAN_VALS = ['no', 'yes']
CONTACT_VALS = ['cellular', 'telephone', 'unknown']
MONTH_VALS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
POUTCOME_VALS = ['failure', 'other', 'success', 'unknown'] # 'nonexistent' no está en bank-full, se usa 'unknown' o 'other'


# Construir la lista de todas las columnas OHE
OHE_COLUMNS = []
for job_val in JOB_VALS: OHE_COLUMNS.append(f'job_{job_val}')
for marital_val in MARITAL_VALS: OHE_COLUMNS.append(f'marital_{marital_val}')
for education_val in EDUCATION_VALS: OHE_COLUMNS.append(f'education_{education_val}')
for default_val in DEFAULT_VALS: OHE_COLUMNS.append(f'default_{default_val}')
for housing_val in HOUSING_VALS: OHE_COLUMNS.append(f'housing_{housing_val}')
for loan_val in LOAN_VALS: OHE_COLUMNS.append(f'loan_{loan_val}')
for contact_val in CONTACT_VALS: OHE_COLUMNS.append(f'contact_{contact_val}')
for month_val in MONTH_VALS: OHE_COLUMNS.append(f'month_{month_val}')
for poutcome_val in POUTCOME_VALS: OHE_COLUMNS.append(f'poutcome_{poutcome_val}')

ALL_MODEL_FEATURES = NUMERICAL_FEATURES_SCALED + OHE_COLUMNS

# --- Cargar Modelo y Preprocesadores ---
@st.cache_resource
def load_keras_model():
    model_path = '../models/bank_marketing_model.keras'
    if not os.path.exists(model_path):
        st.error(f"Error Crítico: El archivo del modelo Keras '{model_path}' no fue encontrado. Asegúrese de que el modelo haya sido entrenado y guardado correctamente.")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo Keras: {e}")
        return None

@st.cache_resource
def load_scaler_object():
    scaler_path = '../models/scaler.joblib'
    if not os.path.exists(scaler_path):
        st.error(f"Error Crítico: El archivo del escalador '{scaler_path}' no fue encontrado. Asegúrese de que el escalador haya sido guardado durante la ingeniería de características.")
        return None
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Error al cargar el objeto Scaler: {e}")
        return None

model = load_keras_model()
scaler = load_scaler_object()

# --- Título de la Aplicación ---
st.title("Predicción de Suscripción a Depósito a Plazo")
st.markdown("Esta aplicación utiliza un modelo de Deep Learning para predecir si un cliente suscribirá un depósito a plazo basándose en sus características y el historial de contacto.")

# --- Entradas del Usuario (Sidebar) ---
st.sidebar.header("Ingrese los Datos del Cliente:")
st.sidebar.info("Por favor, ingrese los datos del cliente en el panel lateral y haga clic en 'Predecir Suscripción'.")

# Numerical Inputs
age = st.sidebar.number_input("Edad", min_value=18, max_value=100, value=40)
balance = st.sidebar.number_input("Saldo Anual Promedio (€)", value=1000, step=100)
day = st.sidebar.number_input("Último Día de Contacto (del mes)", min_value=1, max_value=31, value=15)
duration = st.sidebar.number_input("Duración del Último Contacto (segundos)", value=180, min_value=0, help="Este valor puede influir fuertemente en la predicción.")
campaign = st.sidebar.number_input("Número de Contactos en esta Campaña", value=1, min_value=1)
pdays = st.sidebar.number_input("Días desde el Último Contacto (Campaña Anterior, -1 si no contactado)", value=-1, min_value=-1)
previous = st.sidebar.number_input("Número de Contactos Anteriores", value=0, min_value=0)

# Categorical Inputs
job = st.sidebar.selectbox("Tipo de Trabajo", JOB_VALS, index=JOB_VALS.index('management'))
marital = st.sidebar.selectbox("Estado Civil", MARITAL_VALS, index=MARITAL_VALS.index('married'))
education = st.sidebar.selectbox("Nivel Educativo", EDUCATION_VALS, index=EDUCATION_VALS.index('tertiary'))
default = st.sidebar.selectbox("Crédito en Incumplimiento?", DEFAULT_VALS, index=DEFAULT_VALS.index('no'))
housing = st.sidebar.selectbox("Tiene Préstamo Hipotecario?", HOUSING_VALS, index=HOUSING_VALS.index('yes'))
loan = st.sidebar.selectbox("Tiene Préstamo Personal?", LOAN_VALS, index=LOAN_VALS.index('no'))
contact = st.sidebar.selectbox("Tipo de Contacto", CONTACT_VALS, index=CONTACT_VALS.index('cellular'))
month = st.sidebar.selectbox("Último Mes de Contacto", MONTH_VALS, index=MONTH_VALS.index('may'))
poutcome = st.sidebar.selectbox("Resultado de Campaña Anterior", POUTCOME_VALS, index=POUTCOME_VALS.index('unknown'))


# --- Preprocesamiento de Entradas y Predicción ---
if st.sidebar.button("Predecir Suscripción"):
    if model is None or scaler is None:
        st.error("El modelo o el escalador no se pudieron cargar. Por favor, revise los mensajes de error anteriores y asegúrese de que los archivos necesarios estén en la carpeta '../models/'.")
    else:
        # 1. Recolectar entradas en un diccionario
        input_data = {
            'age': age,
            'balance': balance,
            'day': day,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'poutcome': poutcome
        }
        input_df = pd.DataFrame([input_data])

        # 2. Preprocesamiento
        # Crear DataFrame con todas las columnas esperadas, inicializadas en 0
        processed_df = pd.DataFrame(columns=ALL_MODEL_FEATURES)
        # Llenar con ceros para la única fila que vamos a predecir
        processed_df.loc[0] = 0


        # 2a. One-Hot Encoding
        for cat_col in CATEGORICAL_FEATURES:
            user_value = input_df.iloc[0][cat_col]
            ohe_col_name = f"{cat_col}_{user_value}"
            if ohe_col_name in processed_df.columns:
                processed_df.loc[0, ohe_col_name] = 1
            else:
                # Esto no debería ocurrir si ALL_MODEL_FEATURES está bien definido
                st.warning(f"Advertencia: El valor '{user_value}' para la columna '{cat_col}' generó una columna OHE ('{ohe_col_name}') no esperada por el modelo. Esto puede llevar a errores o predicciones incorrectas.")


        # 2b. Llenar características numéricas originales (antes de escalar)
        for num_col in NUMERICAL_FEATURES_SCALED:
            processed_df.loc[0, num_col] = input_df.iloc[0][num_col]


        # 2c. Escalado de Numéricas
        # Asegurarse de que las columnas numéricas a escalar estén en el orden correcto para el scaler
        numerical_data_to_scale = processed_df.loc[0, NUMERICAL_FEATURES_SCALED].values.reshape(1, -1)
        try:
            scaled_numerical_data = scaler.transform(numerical_data_to_scale)
            # Reemplazar las columnas numéricas originales con las escaladas en processed_df
            processed_df.loc[0, NUMERICAL_FEATURES_SCALED] = scaled_numerical_data[0]
        except Exception as e:
            st.error(f"Error durante el escalado de características numéricas: {e}")
            st.error("Asegúrese de que el escalador cargado ('scaler.joblib') fue ajustado con las mismas características numéricas y en el mismo orden que se listan en NUMERICAL_FEATURES_SCALED.")
            # Detener la ejecución adicional si el escalado falla
            st.stop()


        # 3. Feature Order (ya está garantizado si ALL_MODEL_FEATURES es correcto y se usa para crear processed_df)
        # Solo para verificar, nos aseguramos de que el orden de columnas de processed_df es el mismo que ALL_MODEL_FEATURES
        final_input_df = processed_df[ALL_MODEL_FEATURES]

        # 4. Hacer Predicción
        try:
            prediction_proba = model.predict(final_input_df)
            prediction_value = (prediction_proba[0][0] > 0.5).astype(int) # Probabilidad para la clase 1

            # 5. Mostrar Resultados
            st.subheader("Resultado de la Predicción:")
            if prediction_value == 1:
                st.success("El cliente probablemente SÍ se suscribirá al depósito a plazo.")
            else:
                st.error("El cliente probablemente NO se suscribirá al depósito a plazo.")
            
            st.write(f"Probabilidad de Suscripción (clase 'sí'): {prediction_proba[0][0]:.4f}")

            # Opcional: Mostrar el DataFrame procesado para depuración
            # st.write("DataFrame enviado al modelo (primeras columnas):")
            # st.dataframe(final_input_df.head().iloc[:, :15]) # Mostrar solo algunas para no saturar

        except Exception as e:
            st.error(f"Error durante la predicción: {e}")
            st.error("Esto puede ocurrir si los datos de entrada no coinciden con lo que el modelo espera (número de características, orden, etc.).")
            st.error(f"Forma de los datos enviados al modelo: {final_input_df.shape}")
            st.error(f"Columnas esperadas por el modelo (primeras 10): {ALL_MODEL_FEATURES[:10]}")
            st.error(f"Columnas en los datos enviados (primeras 10): {final_input_df.columns.tolist()[:10]}")

else:
    st.info("La aplicación está lista. Ingrese los datos y haga clic en 'Predecir Suscripción'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado como parte de un proyecto de MLOps.")

# Para ejecutar esta aplicación:
# 1. Asegúrese de tener todas las bibliotecas instaladas (streamlit, pandas, numpy, tensorflow, scikit-learn, joblib).
# 2. Guarde este script como `main.py` en una carpeta, por ejemplo, `app/`.
# 3. Asegúrese de que el modelo entrenado (`bank_marketing_model.keras`) y el scaler (`scaler.joblib`)
#    estén en una carpeta `models/` al mismo nivel que la carpeta `app/` (es decir, `../models/`).
# 4. Abra su terminal, navegue a la carpeta `app/` y ejecute: `streamlit run main.py`
#
# Ejemplo de estructura de carpetas:
# project_root/
# |-- app/
# |   |-- main.py
# |-- models/
# |   |-- bank_marketing_model.keras
# |   |-- scaler.joblib
# |-- notebooks/
# |-- data/
# |-- processed_data/
# ... etc.
#
# El script asume que las columnas CATEGORICAL_FEATURES y NUMERICAL_FEATURES_SCALED,
# así como los valores únicos para OHE (JOB_VALS, etc.) y ALL_MODEL_FEATURES,
# se han definido correctamente para que coincidan con el preprocesamiento exacto
# utilizado durante el entrenamiento del modelo. Cualquier discrepancia aquí resultará en errores.
#
# El escalador (scaler.joblib) debe ser un objeto `sklearn.preprocessing.StandardScaler`
# ajustado (fitted) ÚNICAMENTE con las columnas numéricas en el ORDEN EXACTO definido
# en `NUMERICAL_FEATURES_SCALED` del conjunto de entrenamiento.
#
# El modelo Keras (`bank_marketing_model.keras`) debe haber sido entrenado con un DataFrame
# cuyas columnas estén EXACTAMENTE en el orden definido por `ALL_MODEL_FEATURES`.
```
