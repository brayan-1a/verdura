import pandas as pd
from sklearn.model_selection import train_test_split

# Limpiar los datos
def limpiar_datos(df):
    # Aquí puedes agregar tus pasos de limpieza de datos (eliminar nulos, normalización, etc.)
    df = df.dropna()  # Ejemplo: eliminar filas con valores nulos
    return df

# Dividir los datos en entrenamiento y prueba
def dividir_datos(df):
    X = df[['precio_unitario', 'cantidad_promocion', 'temperatura', 'humedad']]
    y = df['cantidad_vendida']
    
    # Dividir en 80% entrenamiento y 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)
