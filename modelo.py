from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def entrenar_y_evaluar(df):
    """
    Entrena modelo para predecir stock necesario, ajustando hiperparámetros con GridSearchCV.
    
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
    
    # Definir el modelo de Random Forest
    modelo = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Definir el espacio de búsqueda para los hiperparámetros
    parametros = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Usar GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(estimator=modelo, param_grid=parametros, 
                               cv=5, n_jobs=-1, scoring='r2')
    
    # Entrenar el modelo con GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Mejor modelo después de la búsqueda
    modelo_optimo = grid_search.best_estimator_
    
    # Predicciones
    predicciones_train = modelo_optimo.predict(X_train)
    predicciones_test = modelo_optimo.predict(X_test)
    
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
        'cv_scores_mean': grid_search.cv_results_['mean_test_score'].mean(),
        'cv_scores_std': grid_search.cv_results_['std_test_score'].mean()
    }
    
    # Importancia features
    importancia = pd.DataFrame({
        'caracteristica': X.columns,
        'importancia': modelo_optimo.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    return modelo_optimo, resultados, metricas, importancia


















