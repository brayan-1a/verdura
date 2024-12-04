import streamlit as st
import pandas as pd
from data_loading import load_data_from_supabase
from predict import predict_inventory
from model_training import train_random_forest
from config import MODEL_FEATURES

def main():
    st.title('Predicción de Inventario de Verduras')
    
    # Cargar datos directamente desde Supabase
    try:
        data = load_data_from_supabase()
        st.success('Datos cargados correctamente desde Supabase')
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return

    # Opción de entrenar modelo
    if st.button('Entrenar Modelo'):
        model, mse, r2 = train_random_forest(data)
        st.success(f'Modelo entrenado exitosamente')
        st.write(f'Error Cuadrático Medio: {mse}')
        st.write(f'Score R²: {r2}')
    
    # Predicción de inventario
    st.header('Predecir Inventario')
    
    # Formulario para nuevos datos
    new_data = {}
    for feature in MODEL_FEATURES:
        new_data[feature] = st.text_input(f'Ingresa {feature}')
    
    if st.button('Predecir'):
        new_df = pd.DataFrame([new_data])
        prediction = predict_inventory(new_df)  # Realizar predicción
        st.write(f'Inventario Final Predicho: {prediction[0]}')

if __name__ == '__main__':
    main()


