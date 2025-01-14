# preparar_datos.py
import pandas as pd
from conexion import conectar_supabase

def obtener_datos():
    cliente = conectar_supabase()
    # Obtén datos de ventas
    ventas = cliente.table("ventas").select("*").execute().data
    # Convierte a DataFrame
    df_ventas = pd.DataFrame(ventas)
    # Asegúrate de que la columna 'fecha_venta' sea del tipo datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    return df_ventas

def preparar_datos(df_ventas):
    # Aquí no agrupamos los datos por período, solo seleccionamos las columnas relevantes
    df_preparado = df_ventas[['precio_unitario', 'cantidad_promocion', 'temperatura', 'humedad', 'cantidad_vendida']]
    
    # Puedes agregar más pasos de preparación de datos si es necesario (como tratar valores nulos)
    return df_preparado


