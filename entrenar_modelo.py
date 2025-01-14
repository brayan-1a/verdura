import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Entrenamiento del modelo
def entrenar_modelo(df, frecuencia):  # Ahora acepta 'df' y 'frecuencia'
    # Preprocesamiento de los datos
    X = df[['precio_unitario', 'cantidad_promocion', 'temperatura', 'humedad']]  # Características
    y = df['cantidad_vendida']  # Objetivo

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    # Guardar el modelo entrenado
    joblib.dump(modelo, 'modelo.pkl')

    # Evaluación
    predicciones = modelo.predict(X)
    
    # Calcular métricas de error
    mae = mean_absolute_error(y, predicciones)
    mse = mean_squared_error(y, predicciones)
    
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    
    return modelo, mae, mse  # Devuelve el modelo, MAE y MSE

# Predicción del stock
def predecir_stock(precio_unitario, cantidad_promocion, temperatura, humedad):
    # Cargar el modelo entrenado
    modelo = joblib.load('modelo.pkl')

    # Realizar la predicción
    prediccion = modelo.predict([[precio_unitario, cantidad_promocion, temperatura, humedad]])
    return prediccion[0]









