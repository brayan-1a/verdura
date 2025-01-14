import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Entrenar el modelo y devolver el modelo entrenado
def entrenar_modelo():
    # Datos de ejemplo (debes usar tus propios datos en la vida real)
    X = np.array([[2.5, 5, 25, 60], [3.0, 10, 28, 65], [1.5, 3, 22, 55], [4.0, 7, 30, 80], [2.2, 0, 26, 70]])
    y = np.array([100, 150, 80, 200, 50])  # La cantidad de stock que necesitas

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)

    # Evaluación del modelo
    y_pred = modelo_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return modelo_rf, mse

# Función para hacer la predicción del stock (usando el modelo en memoria)
def predecir_stock(modelo, precio_unitario, cantidad_promocion, temperatura, humedad):
    # Preparar los datos de entrada
    datos_entrada = np.array([[precio_unitario, cantidad_promocion, temperatura, humedad]])

    # Realizar la predicción
    prediccion = modelo.predict(datos_entrada)

    return prediccion[0]





