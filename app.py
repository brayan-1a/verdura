import streamlit as st
import pandas as pd
from predict_demand import make_prediction

# Título de la aplicación
st.title("Predicción de Demanda de Productos")

# Ingreso de parámetros
producto = st.selectbox("Selecciona un Producto", ["Producto A", "Producto B", "Producto C"])  # Cambiar según tus productos
descuento = st.number_input("Descuento Aplicado (%)", 0, 100, 0)
temperatura = st.number_input("Temperatura (°C)", -10, 50, 20)
humedad = st.number_input("Humedad (%)", 0, 100, 50)

# Botón para hacer la predicción
if st.button("Predecir Demanda"):
    # Realizar la predicción
    prediccion = make_prediction(producto, descuento, temperatura, humedad)
    st.write(f"La demanda predicha para el producto {producto} es: {prediccion} unidades")









