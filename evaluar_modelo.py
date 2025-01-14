from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

def entrenar_modelo(df, periodo='D'):
    # Convertir 'periodo' a formato numérico
    df['periodo_num'] = pd.to_numeric(df['periodo'].dt.strftime('%Y%m%d'))
    
    X = df[['periodo_num']]
    y = df['demanda']
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar un modelo de Random Forest
    modelo = RandomForestRegressor(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return modelo, mae, mse



