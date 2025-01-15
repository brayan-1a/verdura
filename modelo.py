from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def entrenar_y_evaluar(df):
    """Entrena el modelo con características mejoradas y validación cruzada"""
    
    # Preparar las características (features)
    X = df[['dia_semana', 'mes', 'tendencia', 'es_fin_semana', 'temporada', 'diferencia_inventario', 'cantidad_perdida']].copy()
    
    # Variable objetivo
    y = df['cantidad_vendida']
    
    # Normalizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # Optimización de hiperparámetros con GridSearch
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3]
    }
    
    # Usamos GridSearchCV para encontrar los mejores parámetros
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Entrenamos el modelo con los mejores parámetros
    modelo = grid_search.best_estimator_
    
    # Predicciones
    predicciones_train = modelo.predict(X_train)
    predicciones_test = modelo.predict(X_test)
    
    # Calcular métricas
    rmse_train = np.sqrt(mean_squared_error(y_train, predicciones_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, predicciones_test))
    r2_train = r2_score(y_train, predicciones_train)
    r2_test = r2_score(y_test, predicciones_test)
    
    # Importancia de características
    importancia = pd.DataFrame({
        'caracteristica': X.columns,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    # Crear DataFrame con resultados
    resultados = pd.DataFrame({
        'Valor Real': y_test,
        'Predicción': predicciones_test,
        'Diferencia': abs(y_test - predicciones_test)
    })
    
    # Métricas adicionales
    metricas = {
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'cv_scores_mean': grid_search.best_score_,
        'cv_scores_std': grid_search.cv_results_['std_test_score'].mean()
    }
    
    return modelo, resultados, metricas, importancia

def analizar_errores(resultados):
    """Analiza los errores del modelo en detalle"""
    error_analysis = {
        'error_medio': resultados['Diferencia'].mean(),
        'error_mediano': resultados['Diferencia'].median(),
        'error_std': resultados['Diferencia'].std(),
        'error_max': resultados['Diferencia'].max(),
        'error_min': resultados['Diferencia'].min()
    }
    return error_analysis




















