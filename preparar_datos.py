import pandas as pd
import numpy as np

def preparar_datos_modelo(df_ventas):
    """
    Prepara los datos con características adicionales para mejor predicción
    """
    df = df_ventas.copy()
    df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
    
    # Agrupación diaria
    df_diario = df.groupby(['producto_id', 'fecha_venta']).agg({
        'cantidad_vendida': 'sum',
        'cantidad_perdida': 'sum',
        'inventario_inicial': 'first',
        'inventario_final': 'last'
    }).reset_index()
    
    # Características base
    df_diario['dias_stock'] = df_diario.apply(
        lambda x: x['inventario_final'] / x['cantidad_vendida'] if x['cantidad_vendida'] > 0 else 0,
        axis=1
    )
    df_diario['stock_objetivo'] = df_diario.apply(
        lambda x: max(x['cantidad_vendida'] * 7, x['cantidad_vendida'] + 2 * x.get('variabilidad_ventas', 0)),
        axis=1
    )
    
    # Características temporales
    df_diario['dia_semana'] = df_diario['fecha_venta'].dt.dayofweek
    df_diario['mes'] = df_diario['fecha_venta'].dt.month
    df_diario['es_fin_semana'] = df_diario['dia_semana'].isin([5, 6]).astype(int)
    
    # Nuevas características
    df_diario['ventas_7d'] = df_diario.groupby('producto_id')['cantidad_vendida'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    df_diario['ventas_14d'] = df_diario.groupby('producto_id')['cantidad_vendida'].transform(
        lambda x: x.rolling(window=14, min_periods=1).mean()
    )
    
    df_diario['ventas_fin_semana'] = df_diario.groupby('producto_id').apply(
        lambda x: x[x['es_fin_semana'] == 1]['cantidad_vendida'].mean()
    ).fillna(0)
    
    df_diario['stock_medio'] = (df_diario['inventario_inicial'] + df_diario['inventario_final']) / 2
    
    df_diario['tasa_rotacion'] = df_diario.apply(
        lambda x: x['cantidad_vendida'] / x['stock_medio'] if x['stock_medio'] > 0 else 0,
        axis=1
    )
    
    # Tendencia de ventas (comparación con semana anterior)
    df_diario['tendencia_ventas'] = df_diario.groupby('producto_id')['cantidad_vendida'].transform(
        lambda x: (x - x.shift(7)) / x.shift(7)
    ).fillna(0)
    
    df_diario['tasa_perdida'] = df_diario.apply(
        lambda x: x['cantidad_perdida'] / (x['cantidad_vendida'] + x['cantidad_perdida'])
        if (x['cantidad_vendida'] + x['cantidad_perdida']) > 0 else 0,
        axis=1
    )
    
    df_diario['variabilidad_ventas'] = df_diario.groupby('producto_id')['cantidad_vendida'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )
    
    return df_diario.fillna(0)


















