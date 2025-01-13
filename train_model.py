import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
from preprocess_data import preprocess_data

def train_model():
    # Cargar los datos procesados
    data = preprocess_data()

    # Definir las características (X) y el objetivo (y)
    X = data[['precio_unitario', 'descuento_aplicado', 'temperatura', 'humedad']]  # Características relevantes
    y = data['cantidad_vendida']  # Objetivo

    # Crear y entrenar el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Guardar el modelo entrenado
    with open('modelo_entrenado.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Modelo entrenado y guardado correctamente.")







