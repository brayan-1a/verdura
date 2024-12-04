import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuraciones de Supabase
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Configuraciones de modelo
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

# Parámetros de Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'max_depth': 10
}

# Rutas de directorios
DATA_DIR = 'data/'
MODELS_DIR = 'models/'

# Validar configuraciones
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL y KEY son requeridos. Configure el archivo .env")