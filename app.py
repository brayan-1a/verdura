import streamlit as st
from preparar_datos import obtener_datos, preparar_datos
from entrenar_modelo import entrenar_modelo
from evaluar_modelo import predecir_demanda
import pandas as pd

# Título de la aplicación
st.title("Predicción de Demanda de Productos")

# **Parte 1: Entrenamiento del modelo**
st.header("Entrenar el modelo")

# Selección de período
periodo = st.selectbox("Seleccione el período para el entrenamiento", ["Día", "Semana", "Mes"])
frecuencia = {'Día': 'D', 'Semana': 'W', 'Mes': 'M'}[periodo]

if st.button("Cargar datos y entrenar modelo"):
    # Obtener y preparar los datos
    df_ventas = obtener_datos()
    df_preparado = preparar_datos(df_ventas, frecuencia)
    
    # Entrenar el modelo
    modelo, mae, mse = entrenar_modelo(df_preparado, frecuencia)
    
    # Mostrar las métricas
    st.write(f"**MAE**: {mae}")
    st.write(f"**MSE**: {mse}")
    
    # Almacenar el modelo entrenado para usarlo más tarde
    st.session_state.modelo = modelo
    st.session_state.df_preparado = df_preparado
    st.session_state.frecuencia = frecuencia

# **Parte 2: Hacer predicciones**
if 'modelo' in st.session_state:
    st.header("Hacer predicción de demanda")

    # Selección del producto
    productos = st.session_state.df_preparado['producto_id'].unique()
    producto_seleccionado = st.selectbox("Seleccione el producto", productos)
    
    # Selección de la fecha de predicción
    fechas_futuras = pd.date_range(st.session_state.df_preparado['periodo'].max(), periods=10, freq=st.session_state.frecuencia)
    
    # Selección del período para la predicción
    periodo_prediccion = st.selectbox("Seleccione el período para la predicción", ["Día", "Semana", "Mes"])
    frecuencia_prediccion = {'Día': 'D', 'Semana': 'W', 'Mes': 'M'}[periodo_prediccion]
    
    # Botón de predicción
    if st.button("Predecir demanda"):
        # Filtrar los datos para el producto seleccionado
        df_producto = st.session_state.df_preparado[st.session_state.df_preparado['producto_id'] == producto_seleccionado]
        
        # Predecir la demanda usando el modelo entrenado
        predicciones = predecir_demanda(st.session_state.modelo, fechas_futuras)
        
        # Mostrar los resultados
        resultados = pd.DataFrame({
            "Fecha": fechas_futuras,
            "Demanda Predicha": predicciones
        })
        
        st.write(f"Predicciones para el producto {producto_seleccionado}:")
        st.dataframe(resultados)
else:
    st.warning("Primero entrena el modelo para poder hacer predicciones.")





















