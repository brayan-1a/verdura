from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from data_loading import load_data
from data_preprocessing import preprocess_data

def train_model():
    df = load_data()  # Cargar los datos
    df = preprocess_data(df)  # Preprocesar los datos
    X = df.drop(columns=['cantidad_vendida', 'fecha', 'nombre_cliente', 'dia_semana', 'notas_adicionales'])
    y = df['cantidad_vendida']
    
    # Asegúrate de que X y y no tengan valores nulos
    X = X.fillna(0)  # Si hay NaNs, sustitúyelos con 0 o con el valor adecuado
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear el modelo RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    score = model.score(X_test, y_test)
    print(f"Precisión del modelo: {score}")
    
    return model

