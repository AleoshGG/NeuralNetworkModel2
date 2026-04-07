import tensorflow as tf
from src.entities.data_results import DataResults

class WeightCaptureService(tf.keras.callbacks.Callback):
    """Callback para capturar la evolución de los pesos en cada época."""
    def __init__(self, data_results_entity: DataResults):
        super().__init__()
        self.data_results = data_results_entity

    def on_train_begin(self, logs=None):
        # Capturamos los pesos iniciales (aleatorios antes de entrenar)
        self.data_results.initial_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        # Capturamos los pesos al final de cada época
        self.data_results.add_weight_snapshot(self.model.get_weights())

    def on_train_end(self, logs=None):
        # Capturamos los pesos finales
        self.data_results.final_weights = self.model.get_weights()