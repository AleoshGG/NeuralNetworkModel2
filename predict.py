import os
import numpy as np
import tensorflow as tf


def run_inference():
    # Ruta donde guardamos el modelo en app.py
    model_path = os.path.join('results', 'train', 'xor_model.keras')

    print("Iniciando prueba de predicción...")

    # Validamos que el modelo exista antes de intentar cargarlo
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}.")
        print("Asegúrate de haber ejecutado 'python app.py' para entrenar la red primero.")
        return

    # 1. Cargar el modelo pre-entrenado
    print(f"Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 2. Definir los datos de prueba
    # Usaremos las 4 combinaciones clásicas del XOR para ver cómo responde
    X_test = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    print("\nRealizando predicciones...\n")
    # 3. Obtener la predicción de la red
    # Esto devolverá un arreglo con las probabilidades calculadas por la función sigmoide
    predictions = model.predict(X_test, verbose=0)

    # 4. Formatear y mostrar los resultados
    print("--- Resultados de la Red Neuronal (XOR) ---")
    print(f"{'Entrada (X)':<15} | {'Salida Real':<15} | {'Predicción Cruda':<20} | {'Predicción Final'}")
    print("-" * 75)

    for i in range(len(X_test)):
        entrada = X_test[i]
        # XOR real calculando bit a bit en Python usando el operador ^
        salida_real = entrada[0] ^ entrada[1]

        # El valor crudo es lo que sale directamente de la función sigmoide (ej. 0.9854 o 0.0123)
        valor_crudo = predictions[i][0]

        # Como usamos sigmoide, redondeamos: >= 0.5 es 1, < 0.5 es 0
        prediccion_final = int(round(valor_crudo))

        print(f"{str(entrada):<15} | {salida_real:<15} | {valor_crudo:<20.4f} | {prediccion_final}")


if __name__ == "__main__":
    run_inference()