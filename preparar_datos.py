import pandas as pd

def conectar_supabase():
    """Crear conexión con Supabase"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def obtener_datos():
    """Obtener datos históricos de ventas, inventarios y desperdicio"""
    supabase = conectar_supabase()
    
    # Obtener datos de ventas
    ventas = supabase.table('ventas').select(
        "producto_id,fecha_venta,cantidad_vendida"
    ).execute()
    
    # Obtener datos de inventario
    inventarios = supabase.table('inventarios').select(
        "producto_id,inventario_inicial,inventario_final,fecha_actualizacion"
    ).execute()
    
    # Obtener datos de desperdicio
    desperdicio = supabase.table('desperdicio').select(
        "producto_id,cantidad_perdida,fecha_registro"
    ).execute()
    
    # Convertir a DataFrame
    df_ventas = pd.DataFrame(ventas.data)
    df_inventarios = pd.DataFrame(inventarios.data)
    df_desperdicio = pd.DataFrame(desperdicio.data)
    
    # Merge los datos en un único DataFrame
    df_ventas = df_ventas.merge(df_inventarios, on="producto_id", how="left")
    df_ventas = df_ventas.merge(df_desperdicio, on="producto_id", how="left")
    
    return df_ventas












