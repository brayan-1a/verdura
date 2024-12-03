import pandas as pd

def preprocess_data(df):
    # Asegurarse de que la columna 'fecha' sea un datetime
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
        print("Fecha después de la conversión:", df['fecha'].head())
    else:
        print("Columnas disponibles en el DataFrame:", df.columns)
        raise KeyError("La columna 'fecha' no se encuentra en el DataFrame")
    
    # Extraer día, mes y año de la fecha
    df['dia'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['año'] = df['fecha'].dt.year
    
    # Convertir las variables categóricas a variables dummy
    df = pd.get_dummies(df, columns=['producto', 'proveedor', 'ubicacion', 'metodo_pago', 'condiciones_climaticas', 'nombre_cliente', 
                                      'dia_semana', 'tipo_producto', 'categoria_producto', 'canal_venta'], drop_first=True)
    
    
    return df


