from supabase import create_client, Client
import os
from dotenv import load_dotenv 

# Configuración de conexión a Supabase
url = os.getenv('https://odlosqyzqrggrhvkdovj.supabase.co')  # Asegúrate de configurar estas variables en tu entorno
key = os.getenv('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck')

# Verifica que las variables estén cargadas correctamente
if not url or not key:
    raise ValueError("La URL o la clave de Supabase no están configuradas correctamente.")

# Crear el cliente de Supabase
supabase: Client = create_client(url, key)

def get_data():
    # Obtener datos de las tablas de Supabase (puedes modificar las queries según sea necesario)
    productos = supabase.table('productos').select('*').execute()
    ventas = supabase.table('ventas').select('*').execute()
    promociones = supabase.table('promociones').select('*').execute()
    condiciones_climaticas = supabase.table('condiciones_climaticas').select('*').execute()

    return productos.data, ventas.data, promociones.data, condiciones_climaticas.data
