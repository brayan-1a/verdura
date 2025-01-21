from supabase import create_client
import pandas as pd

# Configuración de Supabase
SUPABASE_URL = "https://odlosqyzqrggrhvkdovj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck"

def conectar_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def obtener_datos():
    supabase = conectar_supabase()
    
    # Obtener datos de ventas
    ventas = supabase.table('ventas').select("*").execute()
    
    # Obtener datos de inventarios
    inventarios = supabase.table('inventarios').select("*").execute()
    
    # Obtener datos de desperdicio
    desperdicio = supabase.table('desperdicio').select("*").execute()
    
    # Obtener datos climáticos
    clima = supabase.table('condiciones_climaticas').select("*").execute()
    
    # Obtener datos de promociones
    promociones = supabase.table('promociones').select("*").execute()
    
    # Convertir a DataFrame
    df_ventas = pd.DataFrame(ventas.data)
    df_inventarios = pd.DataFrame(inventarios.data)
    df_desperdicio = pd.DataFrame(desperdicio.data)
    df_clima = pd.DataFrame(clima.data)
    df_promociones = pd.DataFrame(promociones.data)
    
    # Merge de los datos principales
    df_ventas = df_ventas.merge(df_inventarios[['producto_id', 'inventario_inicial', 
                                               'inventario_final', 'fecha_actualizacion']], 
                               on="producto_id", how="left")
    df_ventas = df_ventas.merge(df_desperdicio[['producto_id', 'cantidad_perdida', 
                                               'fecha_registro']], 
                               on="producto_id", how="left")
    
    # Manejar valores nulos
    df_ventas['inventario_inicial'] = df_ventas['inventario_inicial'].fillna(0)
    df_ventas['inventario_final'] = df_ventas['inventario_final'].fillna(0)
    df_ventas['cantidad_perdida'] = df_ventas['cantidad_perdida'].fillna(0)
    
    return df_ventas, df_clima, df_promociones