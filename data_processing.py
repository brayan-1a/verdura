import pandas as pd
from sklearn.model_selection import train_test_split

# Limpiar los datos eliminando valores nulos
def limpiar_datos(datos):
    return datos.dropna()  # Aquí puedes incluir más pasos de limpieza si es necesario

# Dividir los datos en entrenamiento y prueba
def dividir_datos(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
