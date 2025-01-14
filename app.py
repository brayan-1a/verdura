import streamlit as st
import pandas as pd
from supabase_connector import cargar_productos, cargar_ventas, cargar_inventarios, cargar_desperdicios, cargar_condiciones_climaticas
from modelo import entrenar_random_forest, predecir_stock, entrenar_xgboost
from data_processing import limpiar_datos, dividir_datos
from visualizacion import graficar_ventas, graficar_predicciones
from recomendaciones import calcular_recomendacion_stock

# Título y descripción de la app
st.title('Predicción y Recomendación de Stock para Verduras')
st.write('Este sistema predice el stock necesario para tu negocio de verduras y te recomienda la cantidad de compra.')

# Cargar datos desde Supabase
df_productos = cargar_productos()
df_ventas = cargar_ventas()
df_inventarios = cargar_inventarios()
df_desperdicios = cargar_desperdicios()
df_condiciones_climaticas = cargar_condiciones_climaticas()

# Mostrar los primeros datos de productos para verificar
st.subheader('Datos de Productos')
st.write(df_productos.head())

# Mostrar los primeros datos de ventas para verificar
st.subheader('Datos de Ventas')
st.write(df_ventas.head())

# Preprocesar los datos
datos_limpios = limpiar_datos(df_ventas)  # Aquí puedes limpiar los datos de ventas u otros según sea necesario
X = datos_limpios[['precio_unitario', 'cantidad_promocion', 'temperatura', 'humedad']]  # Ajusta las columnas necesarias
y = datos_limpios['stock_necesario']  # Ajusta la columna que quieres predecir

# Dividir los datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = dividir_datos(X, y)

# Pestañas para navegar entre las secciones
seccion = st.selectbox('Selecciona una acción:', ['Entrenar Modelo', 'Realizar Predicción', 'Ver Gráficos'])

if seccion == 'Entrenar Modelo':
    modelo_opcion = st.selectbox('Selecciona el modelo:', ['Random Forest', 'XGBoost'])
    
    if st.button('Entrenar'):
        if modelo_opcion == 'Random Forest':
            modelo, mse = entrenar_random_forest(X_train, y_train)
        elif modelo_opcion == 'XGBoost':
            modelo, mse = entrenar_xgboost(X_train, y_train)
        
        st.success(f'Modelo entrenado exitosamente. Error cuadrático medio (MSE): {mse}')

elif seccion == 'Realizar Predicción':
    if 'modelo' in locals():  # Asegurarnos de que el modelo ha sido entrenado
        precio_unitario = st.number_input('Precio Unitario del Producto', min_value=0.0)
        cantidad_promocion = st.number_input('Cantidad en Promoción', min_value=0)
        temperatura = st.number_input('Temperatura (°C)', min_value=-50.0)
        humedad = st.number_input('Humedad (%)', min_value=0.0)
        
        cantidad_predicha = predecir_stock(modelo, precio_unitario, cantidad_promocion, temperatura, humedad)
        st.write(f'La cantidad recomendada de stock es: {cantidad_predicha}')
    else:
        st.warning("Primero, entrena el modelo antes de realizar la predicción.")

elif seccion == 'Ver Gráficos':
    st.subheader("Gráficos de Ventas y Predicciones")
    
    # Gráfico de ventas
    graficar_ventas(df_ventas['fecha'], df_ventas['cantidad_vendida'])
    
    # Gráfico de predicciones
    graficar_predicciones([cantidad_predicha])  # Solo un ejemplo de visualización



















