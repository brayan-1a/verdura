import pandas as pd

def preprocess_data(df):
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
        print("Fecha después de la conversión:", df['fecha'].head())
    else:
        print("Columnas disponibles en el DataFrame:", df.columns)
        raise KeyError("La columna 'fecha' no se encuentra en el DataFrame")

    df['dia'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['año'] = df['fecha'].dt.year
    df = pd.get_dummies(df, columns=['producto', 'proveedor', 'ubicacion', 'condiciones_climaticas'], drop_first=True)
    return df


