# entrenar_modelo.py
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Entrenamiento del modelo
def entrenar_modelo(df, frecuencia):
    # Preprocesamiento de los datos (asumiendo que ya se prepara previamente)
    X = df[['precio_unitario', 'cantidad_promocion', 'temperatura', 'humedad']]  # Características
    y = df['cantidad_vendida']  # Objetivo

    # Creación del modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    # Guardar el modelo entrenado
    joblib.dump(modelo, 'modelo.pkl')

    # Evaluación del modelo
    predicciones = modelo.predict(X)
    mse = mean_squared_error(y, predicciones)
    mae = mean_absolute_error(y, predicciones)  # Calculando el MAE también
    
    print(f"Error cuadrático medio (MSE): {mse}")
    print(f"Error absoluto medio (MAE): {mae}")
    
    return modelo, mae, mse










