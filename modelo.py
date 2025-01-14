import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Función de entrenamiento del modelo (opcional si ya tienes un modelo entrenado)
def entrenar_modelo():
    # Aquí iría el código para entrenar el modelo con tus datos
    # Ejemplo simple: precios, promoción, temperatura y humedad como variables predictoras
    X = np.array([[1.0, 10, 25, 60], [1.2, 5, 30, 70], [1.5, 20, 28, 65]])  # Ejemplo de datos
    y = np.array([100, 50, 150])  # Ejemplo de cantidad vendida

    modelo = LinearRegression()
    modelo.fit(X, y)

    # Guardamos el modelo entrenado
    with open('modelo.pkl', 'wb') as f:
        pickle.dump(modelo, f)

# Cargar el modelo entrenado
def cargar_modelo():
    with open('modelo.pkl', 'rb') as f:
        modelo = pickle.load(f)
    return modelo

# Función para hacer la predicción
def predecir_stock(precio_unitario, cantidad_promocion, temperatura, humedad):
    # Cargar el modelo entrenado (si ya está guardado)
    modelo = cargar_modelo()

    # Realizar la predicción
    datos_entrada = np.array([[precio_unitario, cantidad_promocion, temperatura, humedad]])
    prediccion = modelo.predict(datos_entrada)

    return prediccion[0]


