from supabase import create_client, Client
import pandas as pd

# Conectar a Supabase
url = 'https://odlosqyzqrggrhvkdovj.supabase.co'
key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck'
supabase: Client = create_client(url, key)

# Función para cargar productos desde Supabase
def cargar_productos():
    productos = supabase.table('productos').select('*').execute()
    df_productos = pd.DataFrame(productos['data'])
    return df_productos

# Función para cargar ventas desde Supabase
def cargar_ventas():
    ventas = supabase.table('ventas').select('*').execute()
    df_ventas = pd.DataFrame(ventas['data'])
    return df_ventas

# Función para cargar inventarios desde Supabase
def cargar_inventarios():
    inventarios = supabase.table('inventarios').select('*').execute()
    df_inventarios = pd.DataFrame(inventarios['data'])
    return df_inventarios

# Función para cargar desperdicios desde Supabase
def cargar_desperdicios():
    desperdicios = supabase.table('desperdicio').select('*').execute()
    df_desperdicios = pd.DataFrame(desperdicios['data'])
    return df_desperdicios

# Función para cargar condiciones climáticas desde Supabase
def cargar_condiciones_climaticas():
    condiciones = supabase.table('condiciones_climaticas').select('*').execute()
    df_condiciones = pd.DataFrame(condiciones['data'])
    return df_condiciones







