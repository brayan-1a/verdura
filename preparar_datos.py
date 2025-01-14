import pandas as pd

def preparar_datos_modelo(df_ventas):
    """Prepara los datos para el entrenamiento del modelo"""
    
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    
    # Crear características para el modelo
    df_final = df_ventas.copy()
    df_final['dia_semana'] = df_final['fecha_venta'].dt.dayofweek
    df_final['mes'] = df_final['fecha_venta'].dt.month
    
    # Nuevas características:
    df_final['es_festivo'] = df_final['dia_semana'].isin([5, 6]).astype(int)  # 1 si es fin de semana
    df_final['tipo_dia'] = df_final['fecha_venta'].apply(lambda x: 1 if x.weekday() == 0 else 0)  # Día lunes
    df_final['promocion'] = 0  # Añadir columna para promociones (puedes agregar la lógica real aquí)

    # Agrupar ventas por producto, día de la semana y mes
    df_agrupado = df_final.groupby(
        ['producto_id', 'dia_semana', 'mes', 'es_festivo', 'tipo_dia', 'promocion']
    )['cantidad_vendida'].mean().reset_index()
    
    return df_agrupado







