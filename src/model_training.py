from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

from config import MODEL_FEATURES, TARGET_VARIABLE, RF_PARAMS, MODELS_DIR
from data_preprocessing import preprocess_data

def train_random_forest(df):
    """Entrenar modelo Random Forest para predicción de inventario"""
    # Preprocesar datos
    df_processed, scaler = preprocess_data(df)
    
    # Separar características y target
    X = df_processed[MODEL_FEATURES]
    y = df_processed[TARGET_VARIABLE]
    
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inicializar y entrenar modelo
    rf_model = RandomForestRegressor(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    
    # Predecir y evaluar
    y_pred = rf_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    
    # Guardar modelo y scaler
    joblib.dump(rf_model, f'{MODELS_DIR}/random_forest_model.joblib')
    joblib.dump(scaler, f'{MODELS_DIR}/scaler.joblib')
    
    return rf_model, mse, r2





