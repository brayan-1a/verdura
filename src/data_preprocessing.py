import pandas as pd

def preprocess_data(df):
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['dia'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['año'] = df['fecha'].dt.year
    df = pd.get_dummies(df, columns=['producto', 'proveedor', 'ubicacion', 'condiciones_climaticas'], drop_first=True)
    return df
