from supabase import create_client
import pandas as pd
from datetime import datetime, timedelta

# Configuración de Supabase
SUPABASE_URL = "https://odlosqyzqrggrhvkdovj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck"

def conectar_supabase():
    """Crear conexión con Supabase"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def obtener_datos():
    """Obtener datos necesarios para el modelo"""
    supabase = conectar_supabase()
    
    # Obtener ventas con información de productos
    ventas = supabase.table('ventas').select(
        "producto_id,fecha_venta,cantidad_vendida"
    ).execute()
    
    # Obtener inventarios
    inventarios = supabase.table('inventarios').select(
        "producto_id,fecha_actualizacion,inventario_final"
    ).execute()
    
    # Obtener desperdicios
    desperdicios = supabase.table('desperdicio').select(
        "producto_id,fecha_registro,cantidad_perdida"
    ).execute()
    
    # Convertir a DataFrames
    df_ventas = pd.DataFrame(ventas.data)
    df_inventarios = pd.DataFrame(inventarios.data)
    df_desperdicios = pd.DataFrame(desperdicios.data)
    
    return df_ventas, df_inventarios, df_desperdicios







