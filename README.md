# NeuronalNetworkModel2

Este proyecto implementa y entrena una Red Neuronal Artificial para resolver el problema clásico de la compuerta lógica XOR, utilizando Python y TensorFlow/Keras. El proyecto está estructurado con buenas prácticas de ingeniería de software, separando responsabilidades en entidades y servicios.

## Arquitectura del Proyecto

El proyecto sigue una arquitectura orientada a servicios y orientada a objetos, dividiendo la lógica de entrenamiento, evaluación, manipulación de datos y visualización en módulos específicos.

### Estructura de Directorios

```text
NeuronalNetworkModel2/
├── app.py                     # Punto de entrada principal para el entrenamiento y generación de reportes/gráficas.
├── predict.py                 # Script de inferencia para probar el modelo pre-entrenado.
├── datasets/                  # Carpeta de datos.
│   └── xor.csv                # Dataset que contiene las entradas y salidas de la compuerta XOR.
├── results/                   # Artefactos generados durante el entrenamiento.
│   ├── train/xor_model.keras  # Modelo guardado tras el entrenamiento.
│   ├── grafica_error.png      # Gráfica generada de la evolución del error (Loss).
│   ├── grafica_pesos.png      # Gráfica generada de la evolución de los pesos.
│   └── pesos_reporte.csv      # Archivo CSV con el registro de pesos iniciales y finales.
└── src/                       # Código fuente principal.
    ├── neuronal_network.py    # Clase principal que encapsula la construcción y entrenamiento de la red (TensorFlow/Keras).
    ├── entities/
    │   └── data_results.py    # Clase de entidad (DTO) para almacenar historial, métricas y evolución de los pesos.
    └── services/
        ├── load_data_service.py       # Servicio para cargar y separar los datos de entrada y salida desde el CSV.
        ├── plotter_service.py         # Servicio encargado de generar y guardar las gráficas (Matplotlib).
        └── weight_capture_service.py  # Callback personalizado de Keras para capturar los pesos en cada época.
```

### Flujo de Ejecución (Pipeline)
1. **Carga de Datos:** `app.py` utiliza la lógica para leer `datasets/xor.csv` obteniendo los vectores `X` e `y`.
2. **Inicialización:** Se instancia la clase `NeuronalNetwork` definiendo la topología.
3. **Entrenamiento:** Se llama al método `train`. Durante el entrenamiento de Keras, el callback `WeightCaptureService` intercepta el proceso al inicio, al final de cada época y al final del entrenamiento para guardar la evolución de los pesos dentro de la entidad `DataResults`.
4. **Exportación de Resultados:** Una vez concluido el entrenamiento, se guardan el modelo en formato `.keras`, el reporte de pesos en formato `.csv` y se mandan llamar los métodos de `plotter_service.py` para guardar las gráficas de rendimiento.
5. **Inferencia (Opcional):** Ejecutando `predict.py`, se carga el modelo `.keras` guardado y se evalúa con las 4 posibles combinaciones del XOR, imprimiendo el resultado real y la predicción final (redondeada de la función sigmoide).

---

## Arquitectura de la Red Neuronal

La red neuronal está diseñada específicamente para resolver el problema del XOR, el cual es un problema **no linealmente separable**. Por lo tanto, requiere al menos una capa oculta.

La arquitectura se define mediante la API Keras Sequential (`models.Sequential`):

### Topología
- **Capa de Entrada (Input):** 2 dimensiones. Corresponde a los dos valores binarios de entrada de la compuerta XOR ($x_1$, $x_2$).
- **Capa Oculta (Hidden Layer):**
  - **Neuronas:** 4
  - **Función de Activación:** Sigmoide (`sigmoid`).
  - **Propósito:** Extraer características y transformar el espacio de entrada para que las clases sean linealmente separables.
- **Capa de Salida (Output Layer):**
  - **Neuronas:** 1
  - **Función de Activación:** Sigmoide (`sigmoid`).
  - **Propósito:** Producir un valor de salida entre 0 y 1. Durante la predicción, este valor se redondea (>= 0.5 se considera 1, < 0.5 se considera 0).

### Compilación y Optimización
- **Optimizador:** Adam (`tf.keras.optimizers.Adam`). Se utiliza con una tasa de aprendizaje (Learning Rate) alta de `0.1` para asegurar una convergencia rápida.
- **Función de Pérdida (Loss):** Entropía Cruzada Binaria (`binary_crossentropy`). Es la función estándar e ideal para problemas de clasificación binaria (como en este caso, donde la salida es 0 o 1).
- **Métricas:** Exactitud (`accuracy`).

### Parámetros de Entrenamiento
- **Épocas (Epochs):** Configurado por defecto a 150 épocas en `app.py` (aunque la red soporta parámetros dinámicos).
- **Callbacks:** Se utiliza un callback personalizado para guardar el estado de los pesos (matrices de pesos y sesgos/biases de las capas) en cada época y poder visualizar cómo el optimizador los ajusta a lo largo del tiempo.
