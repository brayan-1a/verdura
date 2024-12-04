import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuraciones de Supabase
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Configuración del modelo y parámetros
MODEL_FEATURES = [
    'producto', 
    'inventario_inicial', 
    'cantidad_vendida', 
    'precio', 
    'coste_adquisicion', 
    'ventas_por_hora', 
    'desperdicio', 
    'dia_semana', 
    'mes', 
    'año',
    'promocion',
    'condiciones_climaticas'
]
TARGET_VARIABLE = 'inventario_final'

RF_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'max_depth': 10
}

MODELS_DIR = 'models/'
