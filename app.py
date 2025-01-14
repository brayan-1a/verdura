import streamlit as st
from modelo import entrenar_modelo, predecir_stock

# Título de la aplicación
st.title('Predicción de Stock para Verduras')

# Sección de Entrenamiento del modelo
st.header('Entrenamiento del Modelo')

# Botón para entrenar el modelo
if st.button('Entrenar Modelo'):
    # Entrenar el modelo y obtener el modelo entrenado y MSE
    modelo, mse = entrenar_modelo()
    
    # Guardar el modelo entrenado en la sesión de Streamlit
    st.session_state.modelo = modelo
    
    st.success(f'Modelo entrenado exitosamente. Error cuadrático medio (MSE): {mse:.2f}')
else:
    st.info("Haz clic en el botón para entrenar el modelo.")

# Sección de Predicción
st.header('Realizar Predicción')
precio_unitario = st.number_input('Precio Unitario del Producto', min_value=0.0)
cantidad_promocion = st.number_input('Cantidad en Promoción', min_value=0)
temperatura = st.number_input('Temperatura (°C)', min_value=-50.0, max_value=50.0)
humedad = st.number_input('Humedad (%)', min_value=0, max_value=100)

if st.button('Predecir Stock'):
    # Verifica si el modelo ha sido entrenado y está disponible en la sesión
    if 'modelo' in st.session_state:
        modelo = st.session_state.modelo
        # Realizar la predicción con el modelo cargado desde la sesión
        cantidad_predicha = predecir_stock(modelo, precio_unitario, cantidad_promocion, temperatura, humedad)
        st.write(f'La cantidad recomendada de stock es: {cantidad_predicha}')
    else:
        st.warning("Primero, entrena el modelo antes de realizar la predicción.")
















