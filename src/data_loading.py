from supabase import create_client, Client
import pandas as pd
from config import SUPABASE_URL, SUPABASE_KEY

def load_data():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table('verduras').select('*').execute()
    df = pd.DataFrame(response.data)
    print("Columnas del DataFrame:", df.columns)
    print("Primeras filas del DataFrame:", df.head())
    return df

