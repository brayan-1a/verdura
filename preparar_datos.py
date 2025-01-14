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

def preparar_datos(df_ventas, periodo='D'):
    # Agrupar ventas por período seleccionado (Día, Semana, Mes)
    df_preparado = df_ventas.groupby(pd.Grouper(key='fecha_venta', freq=periodo))['cantidad_vendida'].sum().reset_index()
    df_preparado = df_preparado.rename(columns={'fecha_venta': 'periodo', 'cantidad_vendida': 'demanda'})
    return df_preparado

