import streamlit as st
from train_model import train_model
from predict_demand import make_prediction

# Título de la aplicación
st.title("Predicción de Demanda de Productos")

# Crear una opción de selección para cambiar entre las secciones
page = st.sidebar.radio("Selecciona una opción", ("Entrenar Modelo", "Hacer Predicción"))

if page == "Entrenar Modelo":
    st.header("Entrenar el Modelo de Predicción")
    
    # Botón para entrenar el modelo
    if st.button("Entrenar Modelo"):
        # Ejecutamos la función para entrenar el modelo
        train_model()
        st.success("Modelo entrenado y guardado correctamente.")

elif page == "Hacer Predicción":
    st.header("Hacer una Predicción de Demanda")
    
    # Solo permitir hacer predicción si el modelo ha sido entrenado
    try:
        # Verifica si el modelo existe cargando el archivo
        with open('modelo_entrenado.pkl', 'rb') as f:
            model_loaded = True
    except FileNotFoundError:
        model_loaded = False
    
    if model_loaded:
        # Ingreso de parámetros para la predicción
        producto = st.selectbox("Selecciona un Producto", ["Producto A", "Producto B", "Producto C"])  # Cambiar según tus productos
        descuento = st.number_input("Descuento Aplicado (%)", 0, 100, 0)
        temperatura = st.number_input("Temperatura (°C)", -10, 50, 20)
        humedad = st.number_input("Humedad (%)", 0, 100, 50)

        # Botón para hacer la predicción
        if st.button("Predecir Demanda"):
            # Realizar la predicción
            prediccion = make_prediction(producto, descuento, temperatura, humedad)
            st.write(f"La demanda predicha para el producto {producto} es: {prediccion} unidades")
    else:
        st.warning("Primero debes entrenar el modelo.")










