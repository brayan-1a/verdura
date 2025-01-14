import pandas as pd

def preparar_datos_modelo(df_ventas):
    """Prepara los datos para el entrenamiento del modelo"""
    
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    
    # Crear características para el modelo
    df_final = df_ventas.copy()
    df_final['dia_semana'] = df_final['fecha_venta'].dt.dayofweek
    df_final['mes'] = df_final['fecha_venta'].dt.month
    
    # Agrupar ventas por producto, día de la semana y mes
    df_agrupado = df_final.groupby(
        ['producto_id', 'dia_semana', 'mes']
    )['cantidad_vendida'].mean().reset_index()
    
    return df_agrupado



