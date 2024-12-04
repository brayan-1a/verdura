import joblib
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from config import MODELS_DIR, MODEL_FEATURES

# Cargar el modelo y el scaler desde Supabase
def load_model_from_supabase():
    """Cargar modelo y scaler desde Supabase"""
    
    # URL de los archivos subidos a Supabase
    model_url = 'https://odlosqyzqrggrhvkdovj.supabase.co/storage/v1/object/public/models/random_forest_model.joblib'
    scaler_url = 'https://odlosqyzqrggrhvkdovj.supabase.co/storage/v1/object/public/models/scaler.joblib'

    # Descargar el archivo del modelo
    model_response = requests.get(model_url)
    model = joblib.load(BytesIO(model_response.content))
    
    # Descargar el archivo del scaler
    scaler_response = requests.get(scaler_url)
    scaler = joblib.load(BytesIO(scaler_response.content))
    
    return model, scaler

def predict_inventory(new_data):
    """Predecir inventario para nuevos datos"""
    model, scaler = load_model_from_supabase()  # Cargar modelo y scaler
    
    # Convertir a DataFrame si no lo es
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame([new_data])
    
    # Asegurar que las columnas de entrada están preparadas correctamente
    new_data['promocion'] = new_data['promocion'].astype(int)
    new_data['dia_semana'] = pd.Categorical(new_data['dia_semana']).codes
    new_data['mes'] = pd.Categorical(new_data['mes']).codes
    
    # Escalar las características de entrada usando el scaler
    numeric_features = new_data[MODEL_FEATURES].select_dtypes(include=[np.number]).columns
    new_data[numeric_features] = scaler.transform(new_data[numeric_features])
    
    # Realizar la predicción
    predictions = model.predict(new_data[MODEL_FEATURES])
    
    return predictions



