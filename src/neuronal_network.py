from src.entities.data_results import DataResults
from src.services.weight_capture_service import WeightCaptureService

from tensorflow.keras import layers, models
import tensorflow as tf

class NeuronalNetwork:
    def __init__(self, input_dim=24):
        self.input_dim = input_dim
        self.model = self._build_model()
        self.results = DataResults()

    def _build_model(self):
        """
        Define la arquitectura:
        24 -> 16 (ReLU) -> 8 (ReLU) -> 3 (Softmax)
        """
        model = models.Sequential([
            layers.Dense(16, input_dim=self.input_dim, activation='relu', name="hidden_layer_1"),
            layers.Dense(8, activation='relu', name="hidden_layer_2"),
            layers.Dense(3, activation='softmax', name="output_layer")
        ])

        optimizador_rapido = tf.keras.optimizers.Adam(learning_rate=0.01)

        model.compile(optimizer=optimizador_rapido,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, verbose=0):
        # Instanciamos nuestro capturador de pesos
        weight_service = WeightCaptureService(self.results)
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        # El entrenamiento
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            verbose=verbose,
            callbacks=[weight_service]
        )

        # Guardamos el historial de pérdida y precisión
        self.results.set_history(history)
        return self.results

    def save_model(self, path):
        self.model.save(path)
