import pandas as pd
from datetime import datetime, timedelta

def preparar_datos_modelo(df_ventas, df_inventarios, df_desperdicios):
    """Prepara los datos para el modelo predictivo"""
    
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    df_inventarios['fecha_actualizacion'] = pd.to_datetime(df_inventarios['fecha_actualizacion'])
    df_desperdicios['fecha_registro'] = pd.to_datetime(df_desperdicios['fecha_registro'])
    
    # Agrupar ventas por producto y fecha
    ventas_diarias = df_ventas.groupby(['producto_id', 'fecha_venta'])['cantidad_vendida'].sum().reset_index()
    
    # Agrupar desperdicios por producto y fecha
    desperdicios_diarios = df_desperdicios.groupby(['producto_id', 'fecha_registro'])['cantidad_perdida'].sum().reset_index()
    
    # Combinar ventas y desperdicios
    df_final = pd.merge(
        ventas_diarias,
        desperdicios_diarios,
        left_on=['producto_id', 'fecha_venta'],
        right_on=['producto_id', 'fecha_registro'],
        how='left'
    )
    
    # Llenar valores nulos de desperdicios con 0
    df_final['cantidad_perdida'].fillna(0, inplace=True)
    
    # Crear características adicionales
    df_final['dia_semana'] = df_final['fecha_venta'].dt.dayofweek
    df_final['mes'] = df_final['fecha_venta'].dt.month
    
    return df_final



