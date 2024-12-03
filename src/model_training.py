import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from data_loading import load_data
from data_preprocessing import preprocess_data

def train_model():
    df = load_data()
    df = preprocess_data(df)
    
    # Verificar las columnas antes de eliminar
    columnas_a_eliminar = ['inventario_inicial', 'fecha', 'nombre_cliente', 'dia_semana', 'notas_adicionales']
    columnas_presentes = [col for col in columnas_a_eliminar if col in df.columns]
    
    X = df.drop(columns=columnas_presentes)
    y = df['cantidad_vendida']
    
    # Verificar y convertir todos los datos a numéricos
    print("Tipos de datos antes de convertir:", X.dtypes)
    X = X.apply(pd.to_numeric, errors='coerce')
    print("Tipos de datos después de convertir:", X.dtypes)
    
    # Asegurarse de que no hay NaNs
    if X.isnull().values.any():
        raise ValueError("Hay valores NaNs en el DataFrame después de convertir a numéricos")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Precisión del modelo: {score}")
    return model

if __name__ == "__main__":
    train_model()




