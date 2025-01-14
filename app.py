import streamlit as st
from modelo import predecir_stock  # Importa la función de predicción del modelo

# Título de la aplicación
st.title('Predicción de Stock para Verduras')

# Ingresar datos
precio_unitario = st.number_input('Precio Unitario del Producto', min_value=0.0)
cantidad_promocion = st.number_input('Cantidad en Promoción', min_value=0)
temperatura = st.number_input('Temperatura (°C)', min_value=-50.0, max_value=50.0)
humedad = st.number_input('Humedad (%)', min_value=0, max_value=100)

# Al presionar el botón de predicción
if st.button('Predecir Stock'):
    # Llamar a la función de predicción con los valores ingresados
    cantidad_predicha = predecir_stock(precio_unitario, cantidad_promocion, temperatura, humedad)
    st.write(f'La cantidad recomendada de stock es: {cantidad_predicha}')















