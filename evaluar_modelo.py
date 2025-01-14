def predecir_demanda(modelo, fechas):
    # Convertir fechas a formato numérico
    fechas_num = pd.to_numeric(fechas.dt.strftime('%Y%m%d'))
    predicciones = modelo.predict(fechas_num.values.reshape(-1, 1))
    return predicciones




