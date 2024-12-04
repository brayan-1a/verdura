import joblib
import pandas as pd
import numpy as np
from config import MODELS_DIR, MODEL_FEATURES

def load_model():
    """Cargar modelo entrenado"""
    model = joblib.load(f'{MODELS_DIR}/random_forest_model.joblib')
    label_encoders = joblib.load(f'{MODELS_DIR}/label_encoders.joblib')
    scaler = joblib.load(f'{MODELS_DIR}/scaler.joblib')
    
    return model, label_encoders, scaler

def predict_inventory(new_data):
    """Predecir inventario para nuevos datos"""
    model, label_encoders, scaler = load_model()
    
    # Preprocesar nuevos datos (similar a entrenamiento)
    for col, le in label_encoders.items():
        new_data[col] = le.transform(new_data[col].astype(str))
    
    # Escalar características
    numeric_features = new_data[MODEL_FEATURES].select_dtypes(include=[np.number]).columns
    new_data[numeric_features] = scaler.transform(new_data[numeric_features])
    
    # Predecir
    predictions = model.predict(new_data[MODEL_FEATURES])
    
    return predictions
