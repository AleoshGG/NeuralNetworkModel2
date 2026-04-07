
class DataResults:
    def __init__(self):
        # Evolución del error (loss) y métricas (accuracy)
        self.history = None

        # Pesos: listas de arrays de numpy
        self.initial_weights = []
        self.final_weights = []

        # Evolución de los pesos: una lista donde cada elemento
        # representa los pesos en una época específica
        self.weights_evolution = []

    def set_history(self, history_object):
        """Guarda el objeto history que devuelve model.fit() de TF"""
        self.history = history_object.history

    def add_weight_snapshot(self, weights):
        """Guarda una copia de los pesos actuales de la red"""
        # Usamos copy para asegurar que guardamos el valor, no la referencia
        self.weights_evolution.append([w.copy() for w in weights])