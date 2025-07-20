# HotelReviewSentimentAnalysisLSTM

**Clasificación de opiniones hoteleras usando redes LSTM y embeddings preentrenados**

Este proyecto implementa un modelo de clasificación de texto para predecir la polaridad (positiva/negativa) de reseñas hoteleras. Utiliza una red neuronal recurrente (LSTM) con **embeddings preentrenados** de **Word2Vec** para procesar texto en inglés.

El proyecto demuestra la efectividad de las redes LSTM con embeddings preentrenados para clasificar sentimientos en reseñas de hoteles. A través de un flujo estructurado —desde la gestión de dependencias y preprocesamiento textual (limpieza, tokenización, lematización) hasta el diseño de un modelo bidireccional con Word2Vec— se logró un sistema robusto que alcanza **92% de precisión** en datos de prueba.

### Destacan dos aspectos clave:

- **Optimización técnica**:  
  El uso de embeddings congelados (no entrenables) de Word2Vec aceleró la convergencia y mejoró la generalización, mientras que técnicas como *dropout* recurrente y *early stopping* mitigaron el sobreajuste.

- **Balance interpretativo**:  
  La matriz de confusión y el reporte de clasificación revelan un rendimiento equilibrado entre clases (F1-score: **0.92** para ambas), validando la capacidad del modelo para capturar patrones semánticos complejos en textos cortos.

---

## Características principales

- **Modelo LSTM**: Red bidireccional con 256 unidades para capturar dependencias contextuales en texto.
- **Embeddings preentrenados**: Word2Vec (`word2vec-google-news-300`) para representar semánticamente las palabras.
- **Preprocesamiento robusto**:
  - Conversión a minúsculas
  - Tokenización y lematización
  - Eliminación de *stopwords*, símbolos y números
- **Visualización**: Exploración de longitud de reseñas y balance de clases.
- **Dataset**: `deceptive-opinion.csv` (1,600 reseñas de hoteles etiquetadas como engañosas o genuinas).

---

## Tecnologías utilizadas

- Python  
- TensorFlow / Keras  
- Gensim (Word2Vec)  
- NLTK (procesamiento de texto)  
- Matplotlib (visualización)

---

## Temario del Proyecto

### 1. Configuración Inicial y Dependencias

- Instalación de bibliotecas clave: `numpy`, `tensorflow`, `gensim`, etc.
- Resolución de conflictos de versiones y optimización del entorno.

### 2. Carga y Exploración de Datos

- Obtención del dataset desde GitHub (1,600 reseñas).
- Análisis inicial:
  - Distribución de polaridad (positivo / negativo).
  - Longitud de los textos.

### 3. Preprocesamiento de Texto

- Limpieza de texto:
  - Conversión a minúsculas.
  - Eliminación de símbolos y *stopwords*.
- Tokenización y lematización:
  - Reducción de palabras a su raíz verbal.

### 4. Tokenización y Secuenciación

- Conversión del texto en secuencias numéricas.
- *Padding* para estandarizar la longitud de entradas (500 tokens).

### 5. Embeddings Preentrenados (Word2Vec)

- Carga del modelo preentrenado `word2vec-google-news-300`.
- Construcción de la matriz de embeddings para las palabras del vocabulario.

### 6. Arquitectura del Modelo

- Capa `Embedding` fija con vectores Word2Vec.
- Capa **Bidireccional LSTM** (256 unidades) con `Dropout`.
- Capa de salida sigmoide para clasificación binaria.

### 7. Entrenamiento

- División de datos: 80% entrenamiento / 20% prueba.
- Uso de **Early Stopping** para evitar sobreajuste.

### 8. Evaluación y Resultados

- Reporte de clasificación: precisión, *recall*, F1-score.
- Matriz de confusión.
- Precisión final: **92%**.






