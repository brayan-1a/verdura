import pandas as pd
import numpy as np

def preparar_datos_modelo(df_ventas):
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    
    # Crear características base
    df_preparado = df_ventas.copy()
    df_preparado['dia_semana'] = df_preparado['fecha_venta'].dt.dayofweek
    df_preparado['mes'] = df_preparado['fecha_venta'].dt.month
    df_preparado['es_fin_semana'] = df_preparado['dia_semana'].isin([5, 6]).astype(int)
    df_preparado['temporada'] = pd.cut(df_preparado['mes'], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3])
    
    # Asegurar que todas las columnas necesarias existan
    if 'inventario_inicial' not in df_preparado.columns:
        df_preparado['inventario_inicial'] = 0
    if 'inventario_final' not in df_preparado.columns:
        df_preparado['inventario_final'] = 0
    if 'cantidad_perdida' not in df_preparado.columns:
        df_preparado['cantidad_perdida'] = 0
        
    # Calcular diferencia de inventario
    df_preparado['diferencia_inventario'] = df_preparado['inventario_inicial'] - df_preparado['inventario_final']
    
    # Agregar tendencia
    df_preparado['tendencia'] = np.arange(len(df_preparado))
    
    # Agrupar por producto y fecha
    df_agrupado = df_preparado.groupby(
        ['producto_id', 'fecha_venta']
    ).agg({
        'cantidad_vendida': 'sum',
        'dia_semana': 'first',
        'mes': 'first',
        'tendencia': 'first',
        'es_fin_semana': 'first',
        'temporada': 'first',
        'diferencia_inventario': 'sum',
        'cantidad_perdida': 'sum'
    }).reset_index()
    
    return df_agrupado

















