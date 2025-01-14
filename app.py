import streamlit as st
from preparar_datos import obtener_datos, preparar_datos
from entrenar_modelo import entrenar_modelo
import pandas as pd

st.title("Entrenamiento del Modelo")

# Pestaña para entrenar el modelo
if 'modelo_entrenado' not in st.session_state:
    st.session_state.modelo_entrenado = None

# Solo entrenar el modelo una vez
if st.button("Entrenar Modelo"):
    # Obtener y preparar datos
    df_ventas = obtener_datos()
    
    if df_ventas is None or df_ventas.empty:
        st.error("No se pudieron obtener datos de ventas de Supabase.")
    else:
        # Entrenar el modelo con todos los datos disponibles (sin agrupar)
        try:
            modelo = entrenar_modelo(df_ventas)  # Aquí pasamos los datos sin filtrar ni agrupar
            st.session_state.modelo_entrenado = modelo  # Guardar el modelo entrenado en la sesión
            st.success("Modelo entrenado exitosamente.")
        except Exception as e:
            st.error(f"Hubo un error al entrenar el modelo: {e}")

# Pestaña para hacer predicciones
if st.session_state.modelo_entrenado:
    st.title("Realizar Predicción de Demanda")

    # Selección de período para la predicción (día, semana, mes)
    periodo = st.selectbox("Seleccione el período para la predicción", ["Día", "Semana", "Mes"])
    frecuencia = {'Día': 'D', 'Semana': 'W', 'Mes': 'M'}[periodo]

    # Obtener los datos y preparar según la selección de período
    df_ventas = obtener_datos()
    if df_ventas is None or df_ventas.empty:
        st.error("No se pudieron obtener datos de ventas de Supabase.")
    else:
        # Preparar los datos según la frecuencia seleccionada
        df_preparado = preparar_datos(df_ventas, frecuencia)

        if df_preparado.empty:
            st.error("No se encontraron datos suficientes para la predicción.")
        else:
            st.write("Datos preparados para la predicción:")
            st.dataframe(df_preparado.head())

            # Realizar la predicción
            try:
                # Asumiendo que tienes una función para realizar la predicción
                prediccion = predecir_stock(df_preparado, st.session_state.modelo_entrenado)  
                st.write(f"Predicción de demanda para el período seleccionado: {prediccion}")
            except Exception as e:
                st.error(f"Hubo un error al realizar la predicción: {e}")



























