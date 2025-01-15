import pandas as pd

def preparar_datos_stock(df_ventas, df_inventarios, df_desperdicio, df_promociones, df_condiciones):
    """Prepara los datos para el entrenamiento del modelo de stock"""
    
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    df_inventarios['fecha_actualizacion'] = pd.to_datetime(df_inventarios['fecha_actualizacion'])
    df_desperdicio['fecha_registro'] = pd.to_datetime(df_desperdicio['fecha_registro'])
    df_condiciones['fecha'] = pd.to_datetime(df_condiciones['fecha'])
    
    # Agrupar ventas por producto y fecha
    df_ventas_agrupado = df_ventas.groupby(['producto_id', 'fecha_venta'])['cantidad_vendida'].sum().reset_index()

    # Combinar ventas con inventarios
    df_stock = pd.merge(df_ventas_agrupado, df_inventarios, how='left', left_on=['producto_id', 'fecha_venta'], right_on=['producto_id', 'fecha_actualizacion'])

    # Combinar desperdicio con inventarios
    df_stock = pd.merge(df_stock, df_desperdicio[['producto_id', 'fecha_registro', 'cantidad_perdida']], how='left', left_on=['producto_id', 'fecha_venta'], right_on=['producto_id', 'fecha_registro'])

    # Agregar promociones
    df_stock = pd.merge(df_stock, df_promociones[['producto_id', 'descuento_aplicado', 'fecha_inicio', 'fecha_fin']], how='left', left_on=['producto_id'], right_on=['producto_id'])

    # Agregar condiciones climáticas
    df_stock = pd.merge(df_stock, df_condiciones[['fecha', 'temperatura', 'humedad']], how='left', left_on=['fecha_venta'], right_on=['fecha'])

    # Calcular el stock final (esto será una de las columnas objetivo)
    df_stock['stock_final'] = df_stock['inventario_inicial'] - df_stock['cantidad_vendida'] - df_stock['cantidad_perdida']

    return df_stock










