import joblib
import pandas as pd
import numpy as np
from config import MODELS_DIR, MODEL_FEATURES

def load_model():
    """Cargar modelo entrenado"""
    model = joblib.load(f'{MODELS_DIR}/random_forest_model.joblib')
    scaler = joblib.load(f'{MODELS_DIR}/scaler.joblib')
    
    return model, scaler

def predict_inventory(new_data):
    """Predecir inventario para nuevos datos"""
    model, scaler = load_model()
    
    # Convertir a DataFrame si no lo es
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame([new_data])
    
    # Asegurar tipos de datos
    new_data['promocion'] = new_data['promocion'].astype(int)
    new_data['dia_semana'] = pd.Categorical(new_data['dia_semana']).codes
    new_data['mes'] = pd.Categorical(new_data['mes']).codes
    
    # Escalar características
    numeric_features = new_data[MODEL_FEATURES].select_dtypes(include=[np.number]).columns
    new_data[numeric_features] = scaler.transform(new_data[numeric_features])
    
    # Predecir
    predictions = model.predict(new_data[MODEL_FEATURES])
    
    return predictions
