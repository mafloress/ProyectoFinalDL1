# 1. Usar una imagen base de Python oficial
FROM python:3.9-slim

# 2. Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# 3. Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar los directorios de la aplicación y los modelos
# Asegúrate de que la estructura de directorios en el contenedor coincida con las rutas usadas en la app
COPY ./app /app/app
COPY ./models /app/models
# Si el scaler u otros artefactos están en ./models, esta copia es suficiente.
# Si el EDA o Feature Engineering notebooks son necesarios para ALGO en la app (no deberían serlo), copia notebooks/
# COPY ./notebooks /app/notebooks
# Si datos procesados son necesarios para ALGO en la app (no deberían serlo), copia processed_data/
# COPY ./processed_data /app/processed_data


# 5. Exponer el puerto en el que Streamlit se ejecuta por defecto
EXPOSE 8501

# 6. Comando para ejecutar la aplicación Streamlit cuando el contenedor se inicie
# Usar healthcheck para Streamlit es buena práctica
HEALTHCHECK CMD streamlit hello --server.headless true || exit 1
CMD ["streamlit", "run", "/app/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
