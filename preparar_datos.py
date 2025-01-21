import pandas as pd
import numpy as np

def preparar_datos_modelo(df_ventas):
    """
    Prepara los datos para predicción de stock
    
    Args:
        df_ventas (pd.DataFrame): DataFrame con datos de ventas, inventario y desperdicio
        
    Returns:
        pd.DataFrame: DataFrame preparado para el modelo
    """
    # Validar datos de entrada
    if df_ventas.empty:
        raise ValueError("El DataFrame de ventas está vacío")
        
    df = df_ventas.copy()
    
    # Convertir fechas
    df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
    
    # Agrupar por producto y fecha
    df_diario = df.groupby(['producto_id', 'fecha_venta']).agg({
        'cantidad_vendida': 'sum',
        'cantidad_perdida': 'sum',
        'inventario_inicial': 'first',
        'inventario_final': 'last'
    }).reset_index()
    
    # Calcular métricas de stock
    df_diario['dias_stock'] = df_diario.apply(
        lambda x: x['inventario_final'] / x['cantidad_vendida'] if x['cantidad_vendida'] > 0 else 0,
        axis=1
    )
    df_diario['stock_objetivo'] = df_diario['cantidad_vendida'] * 7  # Stock para 7 días
    
    # Características temporales
    df_diario['dia_semana'] = df_diario['fecha_venta'].dt.dayofweek
    df_diario['mes'] = df_diario['fecha_venta'].dt.month
    df_diario['es_fin_semana'] = df_diario['dia_semana'].isin([5, 6]).astype(int)
    
    # Calcular ventas históricas (últimos 7 días)
    df_diario['ventas_7d'] = df_diario.groupby('producto_id')['cantidad_vendida'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Calcular tasa de pérdida
    df_diario['tasa_perdida'] = df_diario.apply(
        lambda x: x['cantidad_perdida'] / (x['cantidad_vendida'] + x['cantidad_perdida'])
        if (x['cantidad_vendida'] + x['cantidad_perdida']) > 0 else 0,
        axis=1
    )
    
    # Calcular stock de seguridad (basado en variabilidad de ventas)
    df_diario['variabilidad_ventas'] = df_diario.groupby('producto_id')['cantidad_vendida'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )
    
    # Llenar valores NaN con 0
    df_diario = df_diario.fillna(0)
    
    return df_diario


















