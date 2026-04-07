from src.entities.data_results import DataResults
from src.services.weight_capture_service import WeightCaptureService

from tensorflow.keras import layers, models
import tensorflow as tf

class NeuronalNetwork:
    def __init__(self, input_dim=2, hidden_units=4, activation_entity=None):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        # Si no se pasa entidad, usamos sigmoid por defecto para XOR
        self.activation = activation_entity.get_activation() if activation_entity else 'sigmoid'
        self.model = self._build_model()
        self.results = DataResults()

    def _build_model(self):
        """Define la arquitectura: 2 -> 4 (Hidden) -> 1 (Output)"""
        model = models.Sequential([
            # Capa oculta: Necesaria para resolver la no-linealidad del XOR
            layers.Dense(self.hidden_units,
                         input_dim=self.input_dim,
                         activation= self.activation,
                         name="hidden_layer"),
            # Capa de salida: 1 neurona para clasificar 0 o 1
            layers.Dense(1, activation='sigmoid', name="output_layer")
        ])

        optimizador_rapido = tf.keras.optimizers.Adam(learning_rate=0.1)

        model.compile(optimizer=optimizador_rapido,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X, y, epochs=500, verbose=0):
        print(f"Iniciando entrenamiento por {epochs} épocas...")

        # Instanciamos nuestro capturador de pesos
        weight_service = WeightCaptureService(self.results)

        # El entrenamiento
        history = self.model.fit(
            X, y,
            epochs=epochs,
            verbose=verbose,
            callbacks=[weight_service],
            # Usamos una fracción de los datos para validación y ver sobreentrenamiento
            #validation_split=0.1
        )

        # Guardamos el historial de pérdida y precisión
        self.results.set_history(history)
        return self.results

    def save_model(self, path):
        self.model.save(path)