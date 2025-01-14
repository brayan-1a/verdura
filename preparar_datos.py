import pandas as pd

def preparar_datos_modelo(df_ventas):
    """Prepara los datos para el entrenamiento del modelo, incluyendo el stock necesario"""
    
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    
    # Crear características para el modelo
    df_final = df_ventas.copy()
    df_final['dia_semana'] = df_final['fecha_venta'].dt.dayofweek
    df_final['mes'] = df_final['fecha_venta'].dt.month
    
    # Crear variable stock necesario (esto es un ejemplo, ajusta según tu lógica)
    df_final['stock_necesario'] = df_final['cantidad_vendida'] * 1.2  # Asegúrate de ajustar esta lógica
    
    # Agrupar ventas por producto, día de la semana y mes
    df_agrupado = df_final.groupby(
        ['producto_id', 'dia_semana', 'mes']
    )['stock_necesario'].mean().reset_index()
    
    return df_agrupado




