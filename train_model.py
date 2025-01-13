# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from preprocess_data import preprocess_data

def train_model():
    # Preprocesar los datos
    df = preprocess_data()

    # Verifica que los datos estén cargados correctamente
    print(f"Datos preprocesados con {len(df)} registros.")

    # Separar las características (X) y la variable objetivo (y)
    X = df.drop('cantidad_vendida', axis=1)  # Características
    y = df['cantidad_vendida']  # Variable objetivo

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Entrenar el modelo y mostrar información sobre el proceso
    print("Entrenando el modelo...")
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Imprimir las métricas de evaluación
    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"Coeficiente de Determinación (R^2): {r2:.2f}")

    # Guardar el modelo entrenado
    model_filename = 'models/modelo_entrenado.pkl'  # Usar una carpeta específica para el modelo
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)  # Asegura que la carpeta exista
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"Modelo entrenado y guardado correctamente en '{model_filename}'.")

    return model_filename, mse, r2









