from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def entrenar_y_evaluar(df):
    """
    Entrena modelo para predecir stock necesario
    
    Args:
        df (pd.DataFrame): DataFrame preparado con features y target
    
    Returns:
        tuple: (modelo, resultados, métricas, importancia)
    """
    # Validar datos de entrada
    if df.empty:
        raise ValueError("El DataFrame está vacío")
    
    # Features para predecir stock
    features = [
        'ventas_7d',           
        'variabilidad_ventas', 
        'tasa_perdida',        
        'dia_semana',          
        'mes',                 
        'es_fin_semana'        
    ]
    
    # Validar que existan todas las características necesarias
    if any(feature not in df.columns for feature in features):
        raise ValueError(f"Faltan algunas características requeridas: {features}")
    
    X = df[features].copy()
    y = df['stock_objetivo']
    
    # Manejar valores nulos
    X = X.fillna(0)
    y = y.fillna(X['ventas_7d'] * 7)
    
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
        random_state=42,
        n_jobs=-1  # Usar todos los cores disponibles
    )
    
    # Calcular cross validation scores
    cv_scores = cross_val_score(modelo, X_scaled, y, cv=5, scoring='r2')
    
    # Entrenar modelo final
    modelo.fit(X_train, y_train)
    
    # Predicciones
    predicciones_train = modelo.predict(X_train)
    predicciones_test = modelo.predict(X_test)
    
    # Resultados
    resultados = pd.DataFrame({
        'Stock Real': y_test,
        'Stock Predicho': predicciones_test,
        'Diferencia': abs(y_test - predicciones_test)
    })
    
    # Métricas completas
    metricas = {
        'rmse_train': np.sqrt(mean_squared_error(y_train, predicciones_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, predicciones_test)),
        'r2_train': r2_score(y_train, predicciones_train),
        'r2_test': r2_score(y_test, predicciones_test),
        'cv_scores_mean': cv_scores.mean(),
        'cv_scores_std': cv_scores.std()
    }
    
    # Importancia features
    importancia = pd.DataFrame({
        'caracteristica': X.columns,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    return modelo, resultados, metricas, importancia

def analizar_errores(resultados):
    """
    Analiza los errores en las predicciones de stock
    
    Args:
        resultados (pd.DataFrame): DataFrame con predicciones y valores reales
    
    Returns:
        dict: Diccionario con métricas de análisis de errores
    """
    # Validar datos de entrada
    if resultados.empty:
        raise ValueError("El DataFrame de resultados está vacío")
        
    # Calcular métricas de error
    error_medio_unidades = resultados['Diferencia'].mean()
    error_mediano_unidades = resultados['Diferencia'].median()
    maximo_error_unidades = resultados['Diferencia'].max()
    
    # Calcular porcentaje de casos donde el stock predicho fue insuficiente
    stock_insuficiente = (
        (resultados['Stock Predicho'] < resultados['Stock Real']).mean() * 100
    )
    
    return {
        'error_medio_unidades': error_medio_unidades,
        'error_mediano_unidades': error_mediano_unidades,
        'maximo_error_unidades': maximo_error_unidades,
        'stock_insuficiente': stock_insuficiente
    }