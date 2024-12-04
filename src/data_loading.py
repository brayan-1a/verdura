from supabase import create_client
import pandas as pd
from config import SUPABASE_URL, SUPABASE_KEY, MODEL_FEATURES, TARGET_VARIABLE, DATA_DIR

def load_data_from_supabase():
    """Cargar datos desde Supabase"""
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Cargar todos los datos
    response = supabase.table('ventas_verduras').select('*').execute()
    
    # Convertir a DataFrame
    df = pd.DataFrame(response.data)
    
    # Seleccionar solo las características que necesitamos
    df_selected = df[MODEL_FEATURES + [TARGET_VARIABLE]]
    
    # Guardar localmente
    df_selected.to_csv(f'{DATA_DIR}/raw_data.csv', index=False)
    
    return df_selected

def load_local_data():
    """Cargar datos desde archivo local"""
    return pd.read_csv(f'{DATA_DIR}/raw_data.csv')


