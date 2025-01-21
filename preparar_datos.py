import pandas as pd
import numpy as np

def check_promocion_activa(fecha, df_promociones):
    """Verifica si hay una promoción activa para la fecha dada"""
    promociones_activas = df_promociones[
        (df_promociones['fecha_inicio'] <= fecha) & 
        (df_promociones['fecha_fin'] >= fecha)
    ]
    return 1 if not promociones_activas.empty else 0

def calcular_variabilidad_estacional(df):
    """Calcula la variabilidad considerando patrones estacionales"""
    return df.groupby(['producto_id', 'mes'])['cantidad_vendida'].transform(
        lambda x: x.rolling(window=30, min_periods=1).std()
    )

def preparar_datos_modelo(df_ventas, df_clima, df_promociones):
    if df_ventas.empty:
        raise ValueError("El DataFrame de ventas está vacío")
        
    df = df_ventas.copy()
    
    # Convertir fechas
    df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
    
    # Merge con datos climáticos
    df = df.merge(
        df_clima[['fecha', 'temperatura', 'humedad']],
        left_on='fecha_venta',
        right_on='fecha',
        how='left'
    )
    
    # Verificar promociones activas
    df['tiene_promocion'] = df.apply(
        lambda x: check_promocion_activa(x['fecha_venta'], df_promociones),
        axis=1
    )
    
    # Agrupar por producto y fecha
    df_diario = df.groupby(['producto_id', 'fecha_venta']).agg({
        'cantidad_vendida': 'sum',
        'cantidad_perdida': 'sum',
        'inventario_inicial': 'first',
        'inventario_final': 'last',
        'temperatura': 'mean',
        'humedad': 'mean',
        'tiene_promocion': 'max'
    }).reset_index()
    
    # Calcular métricas de stock
    df_diario['dias_stock'] = df_diario.apply(
        lambda x: x['inventario_final'] / x['cantidad_vendida'] if x['cantidad_vendida'] > 0 else 0,
        axis=1
    )
    df_diario['stock_objetivo'] = df_diario['cantidad_vendida'] * 7
    
    # Características temporales
    df_diario['dia_semana'] = df_diario['fecha_venta'].dt.dayofweek
    df_diario['mes'] = df_diario['fecha_venta'].dt.month
    df_diario['es_fin_semana'] = df_diario['dia_semana'].isin([5, 6]).astype(int)
    
    # Ventas históricas y variabilidad
    df_diario['ventas_7d'] = df_diario.groupby('producto_id')['cantidad_vendida'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    df_diario['variabilidad_ventas'] = df_diario.groupby('producto_id')['cantidad_vendida'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )
    
    # Variabilidad estacional
    df_diario['variabilidad_estacional'] = calcular_variabilidad_estacional(df_diario)
    
    # Tasa de pérdida
    df_diario['tasa_perdida'] = df_diario.apply(
        lambda x: x['cantidad_perdida'] / (x['cantidad_vendida'] + x['cantidad_perdida'])
        if (x['cantidad_vendida'] + x['cantidad_perdida']) > 0 else 0,
        axis=1
    )
    
    # Llenar valores NaN
    df_diario = df_diario.fillna(0)
    
    return df_diario


















