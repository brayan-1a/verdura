from supabase import create_client, Client
import pandas as pd

# Conectar a Supabase
url = 'https://odlosqyzqrggrhvkdovj.supabase.co'
key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck'

supabase: Client = create_client(url, key)

# Función para cargar productos desde Supabase
def cargar_productos():
    try:
        # Intentamos obtener los productos
        response = supabase.table('productos').select('*').execute()
        
        # Verificamos si la clave 'data' está en la respuesta
        if 'data' in response:
            df_productos = pd.DataFrame(response['data'])
            return df_productos
        else:
            print("Error: No se encontraron datos en la tabla 'productos'.")
            return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error
    except Exception as e:
        print(f"Error al cargar los productos desde Supabase: {e}")
        return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error







