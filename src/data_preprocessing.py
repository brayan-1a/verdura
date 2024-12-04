import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    """Preprocesar datos para entrenamiento"""
    # Hacer una copia para no modificar el original
    df_processed = df.copy()
    
    # Manejar valores nulos
    imputer = SimpleImputer(strategy='mean')
    
    # Identificar columnas numéricas
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    
    # Imputar valores nulos en columnas numéricas
    df_processed[numeric_columns] = imputer.fit_transform(df_processed[numeric_columns])
    
    # Escalar características numéricas
    scaler = StandardScaler()
    df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
    
    return df_processed, scaler


