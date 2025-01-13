import pandas as pd
from supabase_connector import get_data

def preprocess_data():
    # Obtener los datos de Supabase
    productos, ventas, promociones, condiciones_climaticas = get_data()
    
    # Convertir las listas de diccionarios a DataFrames
    df_productos = pd.DataFrame(productos)
    df_ventas = pd.DataFrame(ventas)
    df_promociones = pd.DataFrame(promociones)
    df_condiciones_climaticas = pd.DataFrame(condiciones_climaticas)
    
    # Verifica las columnas disponibles en df_promociones
    print("Columnas de promociones:", df_promociones.columns)

    # Unir las tablas relevantes
    df = pd.merge(df_ventas, df_productos, on='producto_id', how='left')
    df = pd.merge(df, df_promociones, on='producto_id', how='left')
    df = pd.merge(df, df_condiciones_climaticas, left_on='fecha_venta', right_on='fecha', how='left')
    
    # Limpiar los datos
    # Verifica si 'descuento_aplicado' existe antes de usarlo
    if 'descuento_aplicado' in df.columns:
        df['descuento_aplicado'].fillna(0, inplace=True)
    else:
        print("Advertencia: la columna 'descuento_aplicado' no se encuentra en los datos de promociones.")
        df['descuento_aplicado'] = 0  # Asigna un valor predeterminado si no existe
    
    # Llenar valores nulos en las columnas de clima
    df['temperatura'].fillna(df['temperatura'].mean(), inplace=True)
    df['humedad'].fillna(df['humedad'].mean(), inplace=True)
    
    # Seleccionar solo las columnas necesarias para el modelo
    df = df[['precio_unitario', 'descuento_aplicado', 'temperatura', 'humedad', 'cantidad_vendida']]
    
    return df

