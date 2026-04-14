import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.neuronal_network import NeuronalNetwork
from src.services.load_data_service import load_data_service
from src.services.plotter_service import plot_loss_and_overfitting, plot_weights_evolution

os.makedirs(os.path.join('results', 'train'), exist_ok=True)
os.makedirs(os.path.join('results', 'validated'), exist_ok=True)

def export_weights_to_csv(results, path):
    init_w = np.concatenate([w.flatten() for w in results.initial_weights])
    fin_w = np.concatenate([w.flatten() for w in results.final_weights])

    df = pd.DataFrame({
        'Parametro_ID': [f'Param_{i}' for i in range(len(init_w))],
        'Peso_Inicial': init_w,
        'Peso_Final': fin_w
    })

    df.to_csv(path, index=False)
    print(f"Reporte de pesos guardado en: {path}")

def run_cross_validation(X, y, k=3, epochs=30):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    cv_results = []
    best_model = None
    best_val_loss = float('inf')
    best_history = None
    best_fold = 1
    
    print("\n3 Validación Cruzada")
    print("Se ha aplicado la validación cruzada de los modelos, para")
    print(f"{'Fold':<5} | {'# Obs Entrenamiento':<22} | {'# Obs Validación':<18} | {'Error Entr.':<12} | {'Error Val.':<12} | {'Error Total':<12}")
    print("-" * 95)
    
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        nn = NeuronalNetwork(input_dim=X.shape[1])
        training_results = nn.train(X_train, y_train, X_val, y_val, epochs=epochs, verbose=0)
        
        # Obtenemos los errores de la última época
        train_loss = training_results.history['loss'][-1]
        val_loss = training_results.history['val_loss'][-1]
        total_error = train_loss + val_loss
        
        print(f"{fold:<5} | {len(X_train):<22} | {len(X_val):<18} | {train_loss:<12.4f} | {val_loss:<12.4f} | {total_error:<12.4f}")
        
        cv_results.append({
            'fold': fold,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'total_error': total_error
        })
        
        # Determinar el mejor modelo basado en error de validación
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = nn
            best_history = training_results
            best_fold = fold
            
    print("-" * 95)
    print(f"El mejor modelo es el del Fold {best_fold} con un Error de Validación de {best_val_loss:.4f}.\n")
    return best_model, best_history, best_fold

if __name__ == "__main__":
    dataset_path = os.path.join('datasets', 'Consumer_Shopping_Trends_2026.csv')
    
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el dataset en {dataset_path}")
        exit(1)

    print("Cargando y preprocesando datos...")
    X, y = load_data_service(dataset_path)
    
    best_model, best_history, best_fold = run_cross_validation(X, y, k=3, epochs=30)

    print("4 Entrenamiento")
    print("Para el mejor modelo, presentar la gráfica de evolución del error de entrenamiento")
    print("\n4.1 Evolución del error de entrenamiento")
    print("Gráfica donde se visualicen la evolución del error de entrenamiento para el mejor modelo")
    
    model_path = os.path.join('results', 'train', 'shopping_best_model.keras')
    best_model.save_model(model_path)
    print(f"Mejor modelo guardado en: {model_path}")
    
    # Guardamos el reporte CSV de los pesos del mejor modelo
    csv_path = os.path.join('results', 'pesos_reporte.csv')
    export_weights_to_csv(best_history, csv_path)

    # Generar gráfica de error
    plot_loss_and_overfitting(best_history.history)
    plot_weights_evolution(best_history.weights_evolution)
    
    print("\n¡Gráficas generadas y guardadas en la carpeta /results/!")
    print("\n[ Análisis de Resultados Observados ]")
    print("-> La gráfica 'grafica_error.png' muestra cómo disminuye la pérdida (Loss) tanto en entrenamiento como en validación.")
    print("-> Esto significa que el modelo mejora época tras época y se generaliza bien si la línea roja (validación) baja a la par de la azul (entrenamiento).")
    print("-> La gráfica 'grafica_pesos.png' ilustra la estabilización de los pesos de las neuronas a medida que el modelo aprende.")
