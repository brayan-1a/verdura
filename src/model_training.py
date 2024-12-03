import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from data_loading import load_data
from data_preprocessing import preprocess_data

def train_model():
    df = load_data()
    df = preprocess_data(df)
    X = df.drop(columns=['cantidad_vendida', 'fecha', 'nombre_cliente', 'dia_semana', 'notas_adicionales'])
    y = df['cantidad_vendida']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Precisión del modelo: {score}")
    return model

if __name__ == "__main__":
    train_model()
