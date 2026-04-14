import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

def load_data_service(file_path):
    data = pd.read_csv(file_path)

    # Las variables de entrada (todas menos la última)
    X = data.iloc[:, :-1].copy()
    # La variable de salida (la última columna)
    y_text = data.iloc[:, -1].copy()

    # Identificar y codificar columnas categóricas (texto) a números en X
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Convertir X a numpy array y escalar valores numéricos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convertir Y a numérico y luego a one-hot encoding (3 categorías)
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y_text)
    y_categorical = to_categorical(y_encoded, num_classes=3)

    return X_scaled, y_categorical
