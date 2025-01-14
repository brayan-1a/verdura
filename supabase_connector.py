from supabase import create_client, Client
import pandas as pd

# Conectar a Supabase
url = 'https://odlosqyzqrggrhvkdovj.supabase.co'
key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck'
ssupabase: Client = create_client(url, key)

# Función para cargar productos desde Supabase
def cargar_productos():
    productos = supabase.table('productos').select('*').execute()
    df_productos = pd.DataFrame(productos['data'])
    return df_productos







