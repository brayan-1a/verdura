import streamlit as st
import pandas as pd
import joblib
from data_loading import load_local_data
from predict import predict_inventory
from model_training import train_random_forest

def main():
    st.title('Predicción de Inventario de Verduras')
    
    # Cargar datos
    data = load_local_data()
    
    # Opción de entrenar modelo
    if st.button('Entrenar Modelo'):
        model = train_random_forest(data)
        st.success('Modelo entrenado exitosamente')
    
    # Predicción de inventario
    st.header('Predecir Inventario')
    
    # Formulario para nuevos datos
    new_data = {}
    for feature in MODEL_FEATURES:
        new_data[feature] = st.text_input(f'Ingresa {feature}')
    
    if st.button('Predecir'):
        new_df = pd.DataFrame([new_data])
        prediction = predict_inventory(new_df)
        st.write(f'Inventario Final Predicho: {prediction[0]}')

if __name__ == '__main__':
    main()
