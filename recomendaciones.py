def calcular_recomendacion_stock(prediccion_stock, precio_unitario, costo_adquisicion, desperdicio_estimado):
    # Lógica de recomendación (por ejemplo, comparar el costo de adquisición con el desperdicio)
    recomendacion = prediccion_stock * precio_unitario - desperdicio_estimado
    return recomendacion
