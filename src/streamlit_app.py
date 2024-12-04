import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from config import SUPABASE_URL, SUPABASE_KEY

# Función para cargar los datos de Supabase
def load_data_from_supabase():
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        response = supabase.table('ventas_verduras').select('*').execute()
        df = pd.DataFrame(response.data)
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return pd.DataFrame()

# Función para preprocesar los datos
def preprocess_data(df):
    df_processed = df.copy()
    df_processed['promocion'] = df_processed['promocion'].astype(int)
    df_processed['dia_semana'] = pd.Categorical(df_processed['dia_semana']).codes
    df_processed['mes'] = pd.Categorical(df_processed['mes']).codes
    
    # Escalar los datos numéricos
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
    
    return df_processed, scaler

# Entrenamos el modelo Random Forest
def train_random_forest(df):
    # Preprocesar los datos
    df_processed, scaler = preprocess_data(df)
    
    # Dividir los datos en características (X) y objetivo (y)
    X = df_processed.drop(columns='inventario_final')
    y = df_processed['inventario_final']
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inicializar y entrenar el modelo RandomForest
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2

# Predicción con el modelo entrenado
def predict_inventory(model, scaler, new_data):
    # Preprocesar los nuevos datos
    new_data_processed, _ = preprocess_data(new_data)
    
    # Predecir el inventario final
    predictions = model.predict(new_data_processed)
    
    return predictions

def main():
    st.title('Predicción de Inventario de Verduras')
    
    # Cargar datos desde Supabase
    data = load_data_from_supabase()
    
    if data.empty:
        st.warning("No se pudieron cargar los datos.")
        return
    
    st.write("Datos cargados correctamente")
    st.dataframe(data.head())

    # Entrenamiento del modelo
    if st.button('Entrenar Modelo'):
        model, scaler, mse, r2 = train_random_forest(data)
        st.success('Modelo entrenado exitosamente')
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'R² Score: {r2}')

        # Guardar modelo y scaler
        joblib.dump(model, 'random_forest_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')

        # Graficar resultados
        st.subheader('Gráfica de Predicciones vs Realidad')
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.title('Predicciones vs Valores Reales')
        plt.xlabel('Valor Real')
        plt.ylabel('Valor Predicho')
        st.pyplot()

    # Predicción de inventario
    st.header('Predecir Inventario')

    # Crear formulario para ingresar nuevos datos
    new_data = {}
    for column in data.columns:
        if column != 'inventario_final':  # Excluimos la columna objetivo
            new_data[column] = st.text_input(f'Ingresa {column}')
    
    if st.button('Predecir'):
        new_df = pd.DataFrame([new_data])
        predictions = predict_inventory(model, scaler, new_df)
        st.write(f'Inventario Final Predicho: {predictions[0]}')

if __name__ == '__main__':
    main()



