from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def entrenar_y_evaluar(df):
    """Entrena el modelo y muestra su rendimiento"""
    
    # Preparar features
    X = df[['dia_semana', 'mes']]
    y = df['cantidad_vendida']
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # Entrenar modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Hacer predicciones en conjunto de prueba
    predicciones = modelo.predict(X_test)
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    r2 = r2_score(y_test, predicciones)
    
    # Imprimir resultados
    print("\nResultados del Modelo:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    
    # Crear DataFrame con resultados reales vs predicciones
    resultados = pd.DataFrame({
        'Valor Real': y_test,
        'Predicción': predicciones,
        'Diferencia': abs(y_test - predicciones)
    })
    
    print("\nMuestra de Resultados:")
    print(resultados.head())
    
    return modelo, resultados










