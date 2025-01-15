import pandas as pd

def preparar_datos_modelo(df_ventas, df_inventarios, df_desperdicio):
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    df_inventarios['fecha_actualizacion'] = pd.to_datetime(df_inventarios['fecha_actualizacion'])
    df_desperdicio['fecha_registro'] = pd.to_datetime(df_desperdicio['fecha_registro'])
    
    # Unir los datos de ventas con los de inventarios
    df = pd.merge(df_ventas, df_inventarios, how='left', on='producto_id')
    
    # Unir los datos de desperdicio
    df = pd.merge(df, df_desperdicio, how='left', on='producto_id')
    
    # Crear nuevas características
    df['dia_semana'] = df['fecha_venta'].dt.dayofweek
    df['mes'] = df['fecha_venta'].dt.month
    
    # Rellenar los valores nulos de desperdicio con 0 (si no hay desperdicio registrado)
    df['cantidad_perdida'] = df['cantidad_perdida'].fillna(0)
    
    # Calcular el stock disponible (stock inicial - ventas - desperdicio)
    df['stock_disponible'] = df['inventario_inicial'] - df['cantidad_vendida'] - df['cantidad_perdida']
    
    # Seleccionar características relevantes para el modelo
    df_agrupado = df.groupby(
        ['producto_id', 'dia_semana', 'mes']
    )[['cantidad_vendida', 'stock_disponible']].mean().reset_index()
    
    return df_agrupado











