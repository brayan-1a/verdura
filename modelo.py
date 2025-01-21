from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from supabase import create_client
from conexion import conectar_supabase  # Asumo que la función conectar_supabase está en el archivo conexion.py

def guardar_predicciones_en_supabase(producto, recomendacion_stock, ventas_promedio, tasa_rotacion, tendencia_ventas, tasa_perdida):
    """Guardar las predicciones en la tabla de predicciones de Supabase"""
    # Crear la conexión con Supabase
    supabase = conectar_supabase()
    
    # Datos a insertar
    prediccion_data = {
        "producto": producto,
        "recomendacion_stock": recomendacion_stock,
        "ventas_promedio": ventas_promedio,
        "tasa_rotacion": tasa_rotacion,
        "tendencia_ventas": tendencia_ventas,
        "tasa_perdida": tasa_perdida,
    }
    
    # Insertar en la tabla de predicciones
    supabase.table('predicciones').insert(prediccion_data).execute()

def entrenar_y_evaluar(df):
    """
    Entrena modelo para predecir stock necesario con características adicionales
    """
    # Features originales y nuevos
    features = [
        'ventas_7d',           
        'variabilidad_ventas', 
        'tasa_perdida',        
        'dia_semana',          
        'mes',                 
        'es_fin_semana',
        'ventas_14d',          # Tendencia más larga
        'ventas_fin_semana',   # Patrón de fin de semana
        'stock_medio',         # Stock promedio mantenido
        'tasa_rotacion',       # Velocidad de rotación del stock
        'tendencia_ventas'     # Tendencia de crecimiento/decrecimiento
    ]
    
    X = df[features].copy()
    y = df['stock_objetivo']
    
    # Manejar valores nulos con una estrategia más robusta
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            # Usar la mediana para outliers
            X[col] = X[col].fillna(X[col].median())
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split datos con estratificación por mes para mantener la distribución temporal
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42,
        stratify=df['mes']
    )
    
    # Modelo con hiperparámetros optimizados
    modelo = RandomForestRegressor(
        n_estimators=300,      # Más árboles para mejor generalización
        max_depth=12,          # Profundidad controlada para evitar overfitting
        min_samples_split=5,   # Más muestras para split más robustos
        min_samples_leaf=3,    # Más muestras por hoja para estabilidad
        max_features='sqrt',   # Selección automática de features
        random_state=42,
        n_jobs=-1
    )
    
    # Cross validation con más folds
    cv_scores = cross_val_score(modelo, X_scaled, y, cv=7, scoring='r2')
    
    modelo.fit(X_train, y_train)
    
    predicciones_train = modelo.predict(X_train)
    predicciones_test = modelo.predict(X_test)
    
    # Ajuste de predicciones para reducir stock insuficiente
    factor_seguridad = 1.15  # 15% extra de stock para reducir insuficiencia
    predicciones_test = predicciones_test * factor_seguridad
    
    resultados = pd.DataFrame({
        'Stock Real': y_test,
        'Stock Predicho': predicciones_test,
        'Diferencia': abs(y_test - predicciones_test)
    })
    
    metricas = {
        'rmse_train': np.sqrt(mean_squared_error(y_train, predicciones_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, predicciones_test)),
        'r2_train': r2_score(y_train, predicciones_train),
        'r2_test': r2_score(y_test, predicciones_test),
        'cv_scores_mean': cv_scores.mean(),
        'cv_scores_std': cv_scores.std()
    }
    
    importancia = pd.DataFrame({
        'caracteristica': X.columns,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    # Suponiendo que la predicción es para un producto específico
    producto = "Producto A"  # Este valor lo puedes obtener de las predicciones mismas
    recomendacion_stock = 51  # Este valor es un ejemplo, deberías obtenerlo del modelo
    ventas_promedio = 5.7  # Este valor es un ejemplo, deberías obtenerlo del modelo
    tasa_rotacion = 0.09  # Este valor es un ejemplo, deberías obtenerlo del modelo
    tendencia_ventas = 50.0  # Este valor es un ejemplo, deberías obtenerlo del modelo
    tasa_perdida = 68.4  # Este valor es un ejemplo, deberías obtenerlo del modelo

    # Guardamos las predicciones en la base de datos
    guardar_predicciones_en_supabase(producto, recomendacion_stock, ventas_promedio, tasa_rotacion, tendencia_ventas, tasa_perdida)

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
