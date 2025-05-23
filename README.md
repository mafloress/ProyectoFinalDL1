# Proyecto de Clasificación de Campañas de Marketing Bancario con Deep Learning

## Objetivo del Proyecto

*   Desarrollar un servicio de predicción batch para clasificar el éxito de las campañas de marketing directo de una entidad bancaria utilizando modelos de deep learning.
*   Predecir si un cliente suscribirá un depósito a plazo fijo (`y`).
*   Implementar pipelines de MLOps (características, entrenamiento, inferencia batch).
*   Desplegar el modelo como una aplicación funcional Streamlit y containerizarlo con Docker.

## Descripción del Conjunto de Datos

*   **Nombre:** Bank Marketing Data Set
*   **Fuente:** UCI Machine Learning Repository (originado en una institución bancaria portuguesa).
*   Se utiliza el archivo `bank-full.csv` (45211 instancias, 17 variables de entrada antes del preprocesamiento).
*   **Características principales:** Datos del cliente (edad, trabajo, estado civil, educación), historial de contacto, datos de campaña anterior.
*   **Variable objetivo:** `y` (binaria: 'yes'/'no', indica si el cliente suscribió un depósito a plazo).

## Estructura del Proyecto (Conceptual)

*   `data/`: Contiene los datos crudos (e.g., `bank-full.csv`).
*   `notebooks/`: Jupyter notebooks para análisis exploratorio de datos (EDA) y los diferentes pipelines del proyecto.
    *   `01_eda.ipynb`: Análisis Exploratorio de Datos detallado.
    *   `02_feature_pipeline.ipynb`: Pipeline de Ingeniería de Características (preprocesamiento de datos).
    *   `03_training_pipeline.ipynb`: Pipeline de Entrenamiento del Modelo de Deep Learning con MLflow.
    *   `04_batch_inference_pipeline.ipynb`: Pipeline de Inferencia Batch para generar predicciones sobre nuevos datos.
*   `processed_data/`: Almacena los datos procesados y listos para ser utilizados en el entrenamiento del modelo (e.g., `train_features.csv`, `test_target.csv`).
*   `models/`: Contiene los modelos entrenados y serializados, así como otros artefactos de preprocesamiento (e.g., `bank_marketing_model.keras`, `scaler.joblib`).
*   `app/`: Código fuente de la aplicación web interactiva desarrollada con Streamlit (`main.py`).
*   `predictions/`: Directorio donde se guardan las salidas (predicciones) generadas por el pipeline de inferencia batch.
*   `Dockerfile`: Archivo de configuración para construir la imagen Docker de la aplicación Streamlit.
*   `requirements.txt`: Lista de todas las dependencias de Python necesarias para ejecutar el proyecto.
*   `.github/workflows/`: (Conceptual) Archivos YAML para definir los flujos de trabajo de Integración Continua y Despliegue Continuo (CI/CD) con GitHub Actions.

## Decisiones de Modelado

*   **Modelo Elegido:** Se optó por una Red Neuronal Densa (Perceptrón Multicapa - MLP) implementada con TensorFlow/Keras. Esta elección es adecuada para datos tabulares estructurados como los presentes en el dataset y permite capturar relaciones no lineales complejas entre las características y la variable objetivo.
*   **Preprocesamiento:**
    *   **Valores 'unknown':** Estos valores, presentes en varias columnas categóricas, fueron tratados como una categoría separada durante la codificación one-hot. Esta decisión se tomó porque la categoría 'unknown' podría contener información implícita relevante para la predicción (e.g., el cliente o el banco no desean/pueden proveer la información).
    *   **Variables Categóricas:** Se codificaron utilizando la técnica de One-Hot Encoding (a través de `pd.get_dummies`). Este método crea nuevas columnas binarias para cada categoría, evitando la imposición de un orden ordinal artificial que otros métodos de codificación podrían introducir.
    *   **Variables Numéricas:** Se escalaron utilizando `StandardScaler` de scikit-learn. Este escalador estandariza las características eliminando la media y escalando a la varianza unitaria, lo cual es crucial para el buen rendimiento de las redes neuronales y la convergencia eficiente del optimizador.
*   **Arquitectura del Modelo (Ejemplo Conceptual):**
    *   **Capa de Entrada:** Definida por el número total de características después del preprocesamiento (escalado de numéricas y one-hot encoding de categóricas).
    *   **Capas Ocultas:** Se configuraron 2 capas ocultas densas. La primera con 128 neuronas y la segunda con 64 neuronas. Ambas utilizan la función de activación ReLU (Rectified Linear Unit), elegida por su eficiencia computacional y su capacidad para mitigar el problema del desvanecimiento del gradiente.
    *   **Regularización:** Se aplicaron capas de Dropout después de cada capa oculta (con una tasa de 0.3) para prevenir el sobreajuste (overfitting), mejorando la generalización del modelo a datos no vistos.
    *   **Capa de Salida:** Consiste en una única neurona con función de activación Sigmoide. Esta configuración es estándar para problemas de clasificación binaria, ya que la salida sigmoide puede interpretarse como la probabilidad de pertenencia a la clase positiva (en este caso, la probabilidad de que el cliente suscriba el depósito).
*   **Métricas de Evaluación:** Dada la naturaleza del problema y el posible desbalance de clases en la variable objetivo `y` (más 'no' que 'yes'), se seleccionaron las siguientes métricas para una evaluación completa:
    *   **Accuracy:** Proporción general de predicciones correctas.
    *   **Precisión (Precision):** De todas las predicciones de 'sí', cuántas fueron correctas. Importante para minimizar falsos positivos.
    *   **Recall (Sensibilidad):** De todos los clientes que realmente dirían 'sí', cuántos fueron identificados correctamente. Crucial para minimizar falsos negativos.
    *   **F1-Score:** Media armónica de Precisión y Recall. Es especialmente útil cuando hay un desbalance de clases, ya que busca un equilibrio entre ambas métricas.

## Resultados Principales y Métricas (Ejemplo Conceptual)

*   *Nota Importante: Los siguientes resultados son puramente ilustrativos y conceptuales, ya que el entrenamiento completo y la optimización de hiperparámetros no se ejecutaron como parte de la generación de este esqueleto de proyecto. Los valores reales se obtendrían después de ejecutar el pipeline de entrenamiento (`notebooks/03_training_pipeline.ipynb`).*
*   El modelo de red neuronal, tras el entrenamiento, podría haber alcanzado un **F1-Score de 55.0%** en el conjunto de prueba. (Este valor es un placeholder y dependería del entrenamiento real).
*   **Accuracy:** 90.0%
*   **Precisión (para la clase 'yes'):** 65.0%
*   **Recall (para la clase 'yes'):** 48.0%
*   Los logs detallados del proceso de entrenamiento, incluyendo la pérdida (loss) y las métricas por epoch tanto para el conjunto de entrenamiento como para el de validación, se registraron y pueden ser visualizados utilizando MLflow.

## Instrucciones de Uso

### Configuración del Entorno

1.  **Clonar el Repositorio:**
    ```bash
    # Reemplazar <URL_DEL_REPOSITORIO_AUN_NO_CREADO> con la URL real una vez que el repositorio exista en GitHub.
    git clone <URL_DEL_REPOSITORIO_AUN_NO_CREADO>
    cd nombre-del-repositorio # (e.g., bank_marketing_deep_learning)
    ```

2.  **Crear un Entorno Virtual (Recomendado) y Activar:**
    ```bash
    python -m venv venv
    # En Windows:
    # venv\Scripts\activate
    # En macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

### Ejecución de Notebooks de Jupyter

Se recomienda ejecutar los notebooks ubicados en la carpeta `notebooks/` en el siguiente orden para asegurar la correcta generación de artefactos y datos necesarios para los pasos subsecuentes:

1.  **`01_eda.ipynb`**: Realiza el Análisis Exploratorio de Datos. No genera salidas cruciales para los otros notebooks, pero es fundamental para entender los datos.
2.  **`02_feature_pipeline.ipynb`**: Ejecuta el pipeline de ingeniería de características. Genera los datos procesados y los guarda en la carpeta `processed_data/`. También guarda el objeto `scaler` en `models/`.
3.  **`03_training_pipeline.ipynb`**: Entrena el modelo de deep learning utilizando los datos de `processed_data/`. Guarda el modelo entrenado en la carpeta `models/` y registra el experimento en MLflow.
    *   *Requisito:* Para el registro en MLflow, asegúrese de tener un servidor MLflow ejecutándose o configure MLflow para que registre localmente. Para iniciar la UI de MLflow localmente (desde el directorio raíz del proyecto, donde se creará la carpeta `mlruns`):
        ```bash
        mlflow ui
        ```
4.  **`04_batch_inference_pipeline.ipynb`**: Ejecuta el pipeline de inferencia batch utilizando el modelo guardado y los datos de prueba (o nuevos datos). Genera un archivo de predicciones en la carpeta `predictions/`.

### Ejecución de la Aplicación Streamlit

1.  Asegúrese de que el modelo (`bank_marketing_model.keras`) y el escalador (`scaler.joblib`) existan en la carpeta `models/` (generados por `02_feature_pipeline.ipynb` y `03_training_pipeline.ipynb`).
2.  Desde el directorio raíz del proyecto, ejecute:
    ```bash
    streamlit run app/main.py
    ```
3.  La aplicación estará disponible en su navegador web, generalmente en `http://localhost:8501`.

### Construcción y Ejecución del Contenedor Docker

1.  Asegúrese de tener Docker instalado y en ejecución en su sistema.
2.  Desde el directorio raíz del proyecto (donde se encuentra el `Dockerfile`), construya la imagen Docker:
    ```bash
    docker build -t bank-marketing-app .
    ```
3.  Una vez construida la imagen, ejecute el contenedor:
    ```bash
    docker run -p 8501:8501 bank-marketing-app
    ```
4.  La aplicación Streamlit, ahora containerizada, estará accesible en `http://localhost:8501`.

## Automatización con GitHub Actions (Conceptual)

Para una implementación completa de MLOps, se podrían configurar los siguientes flujos de trabajo (workflows) en la carpeta `.github/workflows/`:

*   **`feature_pipeline.yml` (Pipeline de Características):**
    *   **Activador (Trigger):** Se activa automáticamente con cada `push` a la rama principal (`main` o `master`) que involucre cambios en `notebooks/02_feature_pipeline.ipynb` o en los datos crudos en `data/`.
    *   **Acciones:**
        1.  Configura el entorno de Python e instala las dependencias.
        2.  Ejecuta el notebook `02_feature_pipeline.ipynb`.
        3.  Guarda los artefactos generados (datos en `processed_data/` y el `scaler.joblib` de `models/`) en un sistema de almacenamiento persistente (e.g., Amazon S3, Google Cloud Storage, o como artefactos de GitHub Actions para ser usados por workflows posteriores).

*   **`training_pipeline.yml` (Pipeline de Entrenamiento):**
    *   **Activador (Trigger):** Se activa tras la finalización exitosa del workflow `feature_pipeline.yml` o con cambios directos en `notebooks/03_training_pipeline.ipynb`.
    *   **Acciones:**
        1.  Configura el entorno y dependencias.
        2.  Descarga los datos procesados y el escalador desde el almacenamiento de artefactos.
        3.  Ejecuta el notebook `03_training_pipeline.ipynb` para entrenar el modelo.
        4.  Registra el modelo en MLflow (si está configurado) y guarda el modelo entrenado (`.keras`) como un artefacto en el sistema de almacenamiento.

*   **`inference_pipeline.yml` (Pipeline de Inferencia Batch):**
    *   **Activador (Trigger):** Puede activarse de varias maneras:
        *   En un horario predefinido (e.g., diariamente a las 02:00 AM) usando `schedule cron`.
        *   Manualmente a través de la interfaz de GitHub Actions (`workflow_dispatch`).
        *   Tras la llegada de nuevos datos a un bucket específico (si se integra con servicios de eventos de la nube).
    *   **Acciones:**
        1.  Configura el entorno y dependencias.
        2.  Descarga el último modelo entrenado y el escalador desde el almacenamiento de artefactos.
        3.  Obtiene los datos más recientes que necesitan predicciones (podrían ser de una base de datos, un bucket S3, etc., o usar `test_features.csv` como placeholder).
        4.  Ejecuta el notebook `04_batch_inference_pipeline.ipynb`.
        5.  Guarda las predicciones generadas en `predictions/` en un sistema de almacenamiento o base de datos para su consumo.

## Comentarios Adicionales

*   Todo el código fuente, los notebooks y la documentación de este proyecto han sido desarrollados enteramente en **Español Mexicano**.
*   El proyecto sigue las directrices y mejores prácticas para la operationalización de modelos de Machine Learning, con un enfoque en convertir un análisis exploratorio y modelado inicial (típicamente realizado en notebooks) en un conjunto de pipelines reproducibles y desplegables, culminando en un servicio de predicción batch y una aplicación interactiva.
*   La correcta gestión de las dependencias (`requirements.txt`) y la containerización (`Dockerfile`) aseguran la portabilidad y reproducibilidad del entorno de ejecución.
