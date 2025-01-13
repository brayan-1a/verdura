from supabase import create_client, Client
import os
from dotenv import load_dotenv  # Asegúrate de importar esta función

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la URL y la clave de Supabase desde las variables de entorno
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Verificar si las credenciales están cargadas
if not url or not key:
    raise ValueError("La URL o la clave de Supabase no están configuradas correctamente.")

# Crear el cliente de Supabase
supabase: Client = create_client(url, key)

def get_data():
    # Obtener los datos de Supabase
    productos = supabase.table('productos').select('*').execute()
    ventas = supabase.table('ventas').select('*').execute()
    promociones = supabase.table('promociones').select('*').execute()
    condiciones_climaticas = supabase.table('condiciones_climaticas').select('*').execute()

    return productos.data, ventas.data, promociones.data, condiciones_climaticas.data

