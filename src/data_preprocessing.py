import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from config import MODEL_FEATURES, TARGET_VARIABLE

def preprocess_data(df):
    """Preprocesar datos para entrenamiento"""
    # Hacer una copia para no modificar el original
    df_processed = df.copy()
    
    # Manejar valores nulos
    imputer = SimpleImputer(strategy='mean')
    
    # Encoders para variables categóricas
    label_encoders = {}
    
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            # Encodificar variables categóricas
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
        elif df_processed[col].dtype == 'bool':
            # Convertir booleanos a enteros
            df_processed[col] = df_processed[col].astype(int)
    
    # Imputar valores nulos
    df_processed[MODEL_FEATURES + [TARGET_VARIABLE]] = imputer.fit_transform(
        df_processed[MODEL_FEATURES + [TARGET_VARIABLE]]
    )
    
    # Escalar características numéricas
    scaler = StandardScaler()
    numeric_features = df_processed[MODEL_FEATURES].select_dtypes(include=[np.number]).columns
    df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])
    
    return df_processed, label_encoders, scaler


