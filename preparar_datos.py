import pandas as pd

def preparar_datos_modelo(df_ventas):
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    
    # Crear características para el modelo
    df_ventas['dia_semana'] = df_ventas['fecha_venta'].dt.dayofweek
    df_ventas['mes'] = df_ventas['fecha_venta'].dt.month
    
    # Agrupar datos
    df_agrupado = df_ventas.groupby(
        ['producto_id', 'dia_semana', 'mes']
    )['cantidad_vendida'].mean().reset_index()
    
    return df_agrupado










