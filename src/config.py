import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuraciones de Supabase
SUPABASE_URL = os.getenv('https://odlosqyzqrggrhvkdovj.supabase.co')
SUPABASE_KEY = os.getenv('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck')

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