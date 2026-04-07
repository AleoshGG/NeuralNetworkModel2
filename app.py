import os
import pandas as pd
import numpy as np
from src.neuronal_network import NeuronalNetwork
from src.services.plotter_service import plot_loss_and_overfitting, plot_weights_evolution

# Aseguramos que la estructura de directorios de resultados exista
os.makedirs(os.path.join('results', 'train'), exist_ok=True)
os.makedirs(os.path.join('results', 'validated'), exist_ok=True)


def load_xor_data(file_path):
    """Carga el dataset XOR desde el CSV."""
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values
    return X, y


def export_weights_to_csv(results, path):
    """Genera el dataset CSV con los pesos iniciales y finales."""
    # results.initial_weights es una lista de arrays (matrices de pesos y sesgos de cada capa)
    # Los aplanamos (flatten) para poder ponerlos en columnas de un CSV fácilmente
    init_w = np.concatenate([w.flatten() for w in results.initial_weights])
    fin_w = np.concatenate([w.flatten() for w in results.final_weights])

    df = pd.DataFrame({
        'Parametro_ID': [f'Param_{i}' for i in range(len(init_w))],
        'Peso_Inicial': init_w,
        'Peso_Final': fin_w
    })

    df.to_csv(path, index=False)
    print(f"Reporte de pesos guardado en: {path}")


if __name__ == "__main__":
    dataset_path = os.path.join('datasets', 'xor.csv')

    print("1. Cargando datos...")
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el dataset en {dataset_path}")
        exit(1)

    X, y = load_xor_data(dataset_path)

    print("2. Inicializando la Red Neuronal...")
    # Instanciamos la red con 2 entradas y 4 neuronas ocultas.
    # Como omitimos la clase de activación, la red usa 'sigmoid' por defecto.
    nn = NeuronalNetwork(input_dim=2, hidden_units=4)

    print("3. Entrenando el modelo...")
    # 1000 épocas es un buen número para asegurar que el XOR converja
    training_results = nn.train(X, y, epochs=150, verbose=0)

    # Extraemos el loss final para tener algo de feedback en consola
    final_loss = training_results.history['loss'][-1]
    print(f"Entrenamiento finalizado. Loss final: {final_loss:.4f}")

    print("4. Guardando artefactos...")
    # Guardamos el modelo en formato .keras (el estándar actual de TensorFlow)
    model_path = os.path.join('results', 'train', 'xor_model.keras')
    nn.save_model(model_path)
    print(f"Modelo guardado en: {model_path}")

    # Guardamos el reporte CSV de los pesos
    csv_path = os.path.join('results', 'pesos_reporte.csv')
    export_weights_to_csv(training_results, csv_path)

    print("5. Generando gráficas de rendimiento...")

    # 1. Gráfica de error (Evolución y sobreentrenamiento)
    plot_loss_and_overfitting(training_results.history)

    # 2. Gráfica de evolución de pesos
    plot_weights_evolution(training_results.weights_evolution)

    print("¡Gráficas generadas y guardadas en la carpeta /results/!")