from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def entrenar_modelo_stock(df_stock):
    """Entrena el modelo para predecir el stock final"""
    # Variables predictoras
    X = df_stock[['cantidad_vendida', 'desperdicio', 'temperatura', 'humedad', 'descuento_aplicado']]
    
    # Variable objetivo
    y = df_stock['stock_final']
    
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo RandomForest
    modelo = RandomForestRegressor(n_estimators=200, random_state=42)

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = modelo.predict(X_test)

    # Calcular métricas
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    return modelo, rmse, r2


















