from supabase import create_client
import pandas as pd
import numpy as np
from config import SUPABASE_URL, SUPABASE_KEY, MODEL_FEATURES, TARGET_VARIABLE

def load_data_from_supabase():
    """Cargar datos directamente desde Supabase"""
    try:
        # Crear cliente de Supabase
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Cargar todos los datos
        response = supabase.table('ventas_verduras').select('*').execute()
        
        # Convertir a DataFrame
        df = pd.DataFrame(response.data)
        
        # Seleccionar solo las características que necesitamos
        df_selected = df[MODEL_FEATURES + [TARGET_VARIABLE]].copy()
        
        # Manejo de tipos de datos
        df_selected['promocion'] = df_selected['promocion'].astype(int)
        df_selected['dia_semana'] = pd.Categorical(df_selected['dia_semana']).codes
        df_selected['mes'] = pd.Categorical(df_selected['mes']).codes
        
        return df_selected
    
    except Exception as e:
        print(f"Error al cargar datos de Supabase: {e}")
        raise


