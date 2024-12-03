from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from data_loading import load_data
from data_preprocessing import preprocess_data

def train_model():
    df = load_data()  # Cargar los datos
    df = preprocess_data(df)  # Preprocesar los datos

    # Seleccionar las columnas para las características y la variable dependiente (stock)
    X = df.drop(columns=['inventario_inicial', 'fecha', 'nombre_cliente', 'dia_semana', 'notas_adicionales'])
    y = df['inventario_final']  # Aquí cambiamos 'cantidad_vendida' por 'inventario_final'

    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    score = model.score(X_test, y_test)
    print(f"Precisión del modelo: {score}")
    
    return model


