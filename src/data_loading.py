from supabase import create_client, Client
import pandas as pd
from config import SUPABASE_URL, SUPABASE_KEY

def load_data():
    # Crear el cliente Supabase
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    # Consultar los datos de la tabla 'verduras'
    response = supabase.table('verduras').select('*').execute()
    # Convertir los datos en un DataFrame de pandas
    df = pd.DataFrame(response.data)
    print(df.columns) # Añadir esta línea para verificar las columnas
    return df
