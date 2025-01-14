from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def entrenar_y_evaluar(df):
    """Entrena el modelo para predecir el stock necesario"""
    
    # Preparar características para predicción de stock
    X = df[['dia_semana', 'mes']].copy()
    
    # Agregar características de tendencia temporal
    X['tendencia'] = np.arange(len(X))
    
    # Agregar características estacionales
    X['es_fin_semana'] = X['dia_semana'].isin([5, 6]).astype(int)
    X['temporada'] = pd.cut(X['mes'], bins=[0,3,6,9,12], labels=[0,1,2,3])
    
    # Agregar interacciones
    X['mes_dia'] = X['mes'] * X['dia_semana']
    
    # Variable objetivo: stock necesario (esto depende de la lógica que determines)
    y = df['stock_necesario']  # La variable stock_necesario debe estar en tus datos
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # Crear modelo
    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Validación cruzada
    cv_scores = cross_val_score(modelo, X_scaled, y, cv=5, scoring='r2')
    
    # Entrenar modelo
    modelo.fit(X_train, y_train)
    
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
        'Fecha': df['fecha_venta'],  # Asegúrate de tener esta columna
        'Predicción de Stock Necesario': predicciones_test
    })
    
    # Métricas adicionales
    metricas = {
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'cv_scores_mean': cv_scores.mean(),
        'cv_scores_std': cv_scores.std()
    }
    
    return modelo, resultados, metricas, importancia











