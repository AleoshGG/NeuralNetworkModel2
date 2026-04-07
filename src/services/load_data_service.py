import pandas as pd
import os

def load_data_service(file_path, headers = None):
    # Cargamos el CSV. Si no tiene headers, usamos header=None
    data = pd.read_csv(file_path, header=headers)

    # Las primeras dos columnas son las entradas (X)
    X = data.iloc[:, :2].values
    # La última columna es la salida esperada (y)
    y = data.iloc[:, 2].values

    return X, y

if __name__ == "__main__":
    dataset_path = os.path.join('datasets', 'xor.csv')

    if os.path.exists(dataset_path):
        X, y = load_data_service(dataset_path)
        print("Datos cargados exitosamente:")
        print("Entradas (X):\n", X)
        print("Salidas (y):\n", y)
    else:
        print(f"Error: No se encontró el archivo en {dataset_path}")