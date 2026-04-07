import matplotlib.pyplot as plt
import numpy as np

def plot_loss_and_overfitting(history):
    """
    Muestra la evolución del error y ayuda a detectar sobreentrenamiento
    comparando el Loss de entrenamiento vs el Loss de validación.
    """
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['loss'], 'b-', label='Error de Entrenamiento (Loss)')

    # Solo graficamos validación si existe (por el validation_split)
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r--', label='Error de Validación (Val Loss)')

    plt.title('Evolución del Error y Detección de Sobreentrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (Binary Crossentropy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/grafica_error.png')
    plt.show()


def plot_weights_evolution(weights_evolution):
    """
    Grafica cómo cambian los valores de los pesos a través de las épocas.
    """
    # weights_evolution es una lista de épocas, donde cada época tiene las matrices de pesos.
    # Vamos a aplanar todo para graficar cada parámetro como una línea en el tiempo.

    epochs = len(weights_evolution)
    # Obtenemos la cantidad total de parámetros aplanando la primera época
    num_params = len(np.concatenate([w.flatten() for w in weights_evolution[0]]))

    # Preparamos una matriz para guardar la historia de cada parámetro
    # Filas = épocas, Columnas = cada parámetro individual
    params_history = np.zeros((epochs, num_params))

    for epoch in range(epochs):
        # Aplanamos los pesos de la época actual y los guardamos en su fila
        flat_weights = np.concatenate([w.flatten() for w in weights_evolution[epoch]])
        params_history[epoch, :] = flat_weights

    plt.figure(figsize=(12, 6))

    # Graficamos una línea por cada parámetro
    for i in range(num_params):
        plt.plot(range(1, epochs + 1), params_history[:, i], alpha=0.6)

    plt.title('Evolución de los Pesos de la Red a lo largo del tiempo')
    plt.xlabel('Épocas')
    plt.ylabel('Valor del Peso')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/grafica_pesos.png')
    plt.show()