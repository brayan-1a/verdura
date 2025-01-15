from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def entrenar_y_evaluar(df):
    """Entrena modelo para predecir stock necesario"""
    
    # Features para predecir stock
    features = [
        'ventas_7d',           # Promedio de ventas últimos 7 días
        'variabilidad_ventas', # Variabilidad en ventas
        'tasa_perdida',        # Tasa de pérdida histórica
        'dia_semana',          # Patrón semanal
        'mes',                 # Patrón mensual
        'es_fin_semana'        # Efecto fin de semana
    ]
    
    X = df[features].copy()
    y = df['stock_objetivo']  # Objetivo: stock necesario para 7 días
    
    # Manejar valores nulos
    X = X.fillna(0)
    y = y.fillna(X['ventas_7d'] * 7)  # Stock base si no hay histórico
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Entrenar modelo
    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    
    modelo.fit(X_train, y_train)
    
    # Predicciones
    predicciones_test = modelo.predict(X_test)
    
    # Resultados
    resultados = pd.DataFrame({
        'Stock Real': y_test,
        'Stock Predicho': predicciones_test,
        'Diferencia': abs(y_test - predicciones_test)
    })
    
    # Métricas
    metricas = {
        'rmse_test': np.sqrt(mean_squared_error(y_test, predicciones_test)),
        'r2_test': r2_score(y_test, predicciones_test),
    }
    
    # Importancia features
    importancia = pd.DataFrame({
        'caracteristica': X.columns,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    return modelo, resultados, metricas, importancia

def analizar_errores(resultados):
    """Analiza errores en predicción de stock"""
    error_analysis = {
        'error_medio_unidades': resultados['Diferencia'].mean(),
        'error_mediano_unidades': resultados['Diferencia'].median(),
        'maximo_error_unidades': resultados['Diferencia'].max(),
        'stock_insuficiente': (resultados['Stock Predicho'] < resultados['Stock Real']).mean() * 100
    }
    return error_analysis

















