from supabase import create_client
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# Configuración de Supabase
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

def conectar_supabase():
    """Crear conexión con Supabase"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def obtener_datos():
    """Obtener datos históricos de ventas, inventarios y desperdicio"""
    supabase = conectar_supabase()
    
    # Obtener datos de ventas
    ventas = supabase.table('ventas').select(
        "producto_id, fecha_venta, cantidad_vendida"
    ).execute()
    
    # Obtener datos de inventarios
    inventarios = supabase.table('inventarios').select(
        "producto_id, inventario_inicial, inventario_final, fecha_actualizacion"
    ).execute()
    
    # Obtener datos de desperdicio
    desperdicio = supabase.table('desperdicio').select(
        "producto_id, cantidad_perdida, fecha_registro"
    ).execute()
    
    # Convertir a DataFrame
    df_ventas = pd.DataFrame(ventas.data)
    df_inventarios = pd.DataFrame(inventarios.data)
    df_desperdicio = pd.DataFrame(desperdicio.data)
    
    # Verificar si hay datos de inventarios y desperdicio
    if df_inventarios.empty:
        print("Advertencia: No se encontraron datos de inventarios.")
        df_inventarios = pd.DataFrame(columns=["producto_id", "inventario_inicial", 
                                             "inventario_final", "fecha_actualizacion"])

    if df_desperdicio.empty:
        print("Advertencia: No se encontraron datos de desperdicio.")
        df_desperdicio = pd.DataFrame(columns=["producto_id", "cantidad_perdida", 
                                             "fecha_registro"])

    # Merge de los datos en un único DataFrame
    df_ventas = df_ventas.merge(df_inventarios, on="producto_id", how="left")
    df_ventas = df_ventas.merge(df_desperdicio, on="producto_id", how="left")
    
    # Asegurar que no haya valores nulos en las columnas clave
    df_ventas['inventario_inicial'] = df_ventas['inventario_inicial'].fillna(0)
    df_ventas['inventario_final'] = df_ventas['inventario_final'].fillna(0)
    df_ventas['cantidad_perdida'] = df_ventas['cantidad_perdida'].fillna(0)
    
    return df_ventas