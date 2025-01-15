from supabase import create_client
import pandas as pd

# Configuración de Supabase
SUPABASE_URL = "https://odlosqyzqrggrhvkdovj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck"

def conectar_supabase():
    """Crear conexión con Supabase"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def obtener_datos_stock():
    """Obtener datos relevantes para predecir el stock"""
    supabase = conectar_supabase()

    # Obtener ventas, inventarios, desperdicio, promociones y condiciones climáticas
    ventas = supabase.table('ventas').select(
        "producto_id, fecha_venta, cantidad_vendida"
    ).execute()

    inventarios = supabase.table('inventarios').select(
        "producto_id, fecha_actualizacion, inventario_inicial, inventario_final"
    ).execute()

    desperdicio = supabase.table('desperdicio').select(
        "producto_id, cantidad_perdida, fecha_registro"
    ).execute()

    promociones = supabase.table('promociones').select(
        "producto_id, descuento_aplicado, fecha_inicio, fecha_fin"
    ).execute()

    condiciones_climaticas = supabase.table('condiciones_climaticas').select(
        "fecha, temperatura, humedad"
    ).execute()

    # Convertir a DataFrame
    df_ventas = pd.DataFrame(ventas.data)
    df_inventarios = pd.DataFrame(inventarios.data)
    df_desperdicio = pd.DataFrame(desperdicio.data)
    df_promociones = pd.DataFrame(promociones.data)
    df_condiciones = pd.DataFrame(condiciones_climaticas.data)

    return df_ventas, df_inventarios, df_desperdicio, df_promociones, df_condiciones







