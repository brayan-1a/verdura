from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def entrenar_modelo(df_final):
    """Entrena el modelo predictivo"""
    
    # Preparar features
    features = ['dia_semana', 'mes', 'cantidad_perdida']
    X = df_final[features]
    y = df_final['cantidad_vendida']
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Evaluar el modelo
    predicciones = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    r2 = r2_score(y_test, predicciones)
    
    return modelo, rmse, r2

def predecir_stock(modelo, datos_nuevos):
    """Realiza predicciones de stock necesario"""
    return modelo.predict(datos_nuevos)










