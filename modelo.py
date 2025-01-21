from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

def entrenar_y_evaluar(df):
    features = [
        'ventas_7d',
        'variabilidad_ventas',
        'variabilidad_estacional',
        'tasa_perdida',
        'dia_semana',
        'mes',
        'es_fin_semana',
        'temperatura',
        'humedad',
        'tiene_promocion'
    ]
    
    if any(feature not in df.columns for feature in features):
        raise ValueError(f"Faltan algunas características requeridas: {features}")
    
    X = df[features].copy()
    y = df['stock_objetivo']
    
    X = X.fillna(0)
    y = y.fillna(X['ventas_7d'] * 7)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    modelo = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(modelo, X_scaled, y, cv=5, scoring='r2')
    
    modelo.fit(X_train, y_train)
    
    predicciones_train = modelo.predict(X_train)
    predicciones_test = modelo.predict(X_test)
    
    resultados = pd.DataFrame({
        'Stock Real': y_test,
        'Stock Predicho': predicciones_test,
        'Diferencia': abs(y_test - predicciones_test),
        'dia_semana': X_test['dia_semana'],
        'mes': X_test['mes']
    })
    
    metricas = {
        'rmse_train': np.sqrt(mean_squared_error(y_train, predicciones_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, predicciones_test)),
        'r2_train': r2_score(y_train, predicciones_train),
        'r2_test': r2_score(y_test, predicciones_test),
        'cv_scores_mean': cv_scores.mean(),
        'cv_scores_std': cv_scores.std(),
        'mae': mean_absolute_error(y_test, predicciones_test)
    }
    
    importancia = pd.DataFrame({
        'caracteristica': X.columns,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    return modelo, resultados, metricas, importancia

def analizar_errores(resultados):
    error_medio_unidades = resultados['Diferencia'].mean()
    error_mediano_unidades = resultados['Diferencia'].median()
    maximo_error_unidades = resultados['Diferencia'].max()
    
    stock_insuficiente = (
        (resultados['Stock Predicho'] < resultados['Stock Real']).mean() * 100
    )
    
    # Análisis por día de la semana
    error_por_dia = resultados.groupby('dia_semana')['Diferencia'].mean()
    
    # Análisis por mes
    error_por_mes = resultados.groupby('mes')['Diferencia'].mean()
    
    return {
        'error_medio_unidades': error_medio_unidades,
        'error_mediano_unidades': error_mediano_unidades,
        'maximo_error_unidades': maximo_error_unidades,
        'stock_insuficiente': stock_insuficiente,
        'error_por_dia': error_por_dia,
        'error_por_mes': error_por_mes
    }

def ajustar_prediccion_stock(prediccion_base, error_historico):
    factor_seguridad = 1.2
    factor_error = 0.5 * error_historico
    
    prediccion_ajustada = prediccion_base * factor_seguridad + factor_error
    return prediccion_ajustada