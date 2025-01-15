import pandas as pd

def preparar_datos_modelo(df_ventas):
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    
    # Crear características para el modelo
    df_ventas['dia_semana'] = df_ventas['fecha_venta'].dt.dayofweek
    df_ventas['mes'] = df_ventas['fecha_venta'].dt.month
    df_ventas['año'] = df_ventas['fecha_venta'].dt.year
    
    # Añadir tendencia temporal (por producto)
    df_ventas['tendencia'] = df_ventas.groupby('producto_id').cumcount()
    
    # Estacionalidad
    df_ventas['es_fin_semana'] = df_ventas['dia_semana'].isin([5, 6]).astype(int)
    df_ventas['temporada'] = pd.cut(df_ventas['mes'], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3])
    
    # Agregar diferencia de inventario (inicial - final)
    df_ventas['diferencia_inventario'] = df_ventas['inventario_inicial'] - df_ventas['inventario_final']
    
    # Agregar cantidad perdida
    df_ventas['cantidad_perdida'] = df_ventas['cantidad_perdida'].fillna(0)  # Si no hay pérdida, ponemos 0
    
    # Agrupar datos por producto, día de la semana y mes
    df_agrupado = df_ventas.groupby(
        ['producto_id', 'dia_semana', 'mes', 'año']
    )['cantidad_vendida'].mean().reset_index()

    # Normalizar las características
    df_agrupado['cantidad_vendida'] = (df_agrupado['cantidad_vendida'] - df_agrupado['cantidad_vendida'].mean()) / df_agrupado['cantidad_vendida'].std()

    return df_agrupado













