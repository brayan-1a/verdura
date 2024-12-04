from supabase import create_client
import pandas as pd
from config import SUPABASE_URL, SUPABASE_KEY, MODEL_FEATURES, TARGET_VARIABLE

def load_data_from_supabase():
    """Cargar datos directamente desde Supabase"""
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Cargar todos los datos
    response = supabase.table('ventas_verduras').select('*').execute()
    
    # Convertir a DataFrame
    df = pd.DataFrame(response.data)
    
    # Seleccionar solo las características que necesitamos
    df_selected = df[MODEL_FEATURES + [TARGET_VARIABLE]]
    
    return df_selected


