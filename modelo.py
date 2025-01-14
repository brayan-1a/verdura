from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle

# Entrenar un modelo de Random Forest
def entrenar_random_forest(X_train, y_train):
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    
    # Guardar el modelo entrenado temporalmente
    with open('modelo.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    return modelo, mse

# Entrenar un modelo XGBoost
def entrenar_xgboost(X_train, y_train):
    modelo = xgb.XGBRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    
    # Guardar el modelo entrenado temporalmente
    with open('modelo.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    return modelo, mse

# Cargar el modelo entrenado
def cargar_modelo():
    with open('modelo.pkl', 'rb') as f:
        modelo = pickle.load(f)
    return modelo

# Realizar una predicción de stock
def predecir_stock(modelo, precio_unitario, cantidad_promocion, temperatura, humedad):
    X_input = [[precio_unitario, cantidad_promocion, temperatura, humedad]]
    cantidad_predicha = modelo.predict(X_input)
    return cantidad_predicha[0]






