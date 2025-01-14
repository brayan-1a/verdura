import streamlit as st
from preparar_datos import obtener_datos, preparar_datos
from entrenar_modelo import entrenar_modelo
import pandas as pd

st.title("Entrenamiento del Modelo")

# Pestaña para entrenar el modelo
if 'modelo_entrenado' not in st.session_state:
    st.session_state.modelo_entrenado = None

# Selección de período para entrenar el modelo
periodo = st.selectbox("Seleccione el período para el entrenamiento", ["Día", "Semana", "Mes"])
frecuencia = {'Día': 'D', 'Semana': 'W', 'Mes': 'M'}[periodo]

if st.button("Entrenar Modelo"):
    # Obtener y preparar datos
    df_ventas = obtener_datos()
    
    if df_ventas is None or df_ventas.empty:
        st.error("No se pudieron obtener datos de ventas de Supabase.")
    else:
        df_preparado = preparar_datos(df_ventas, frecuencia)

        if df_preparado.empty:
            st.error("No se encontraron datos suficientes para entrenar el modelo.")
        else:
            # Verificar si los datos están en el formato correcto
            st.write("Datos preparados para entrenamiento:")
            st.dataframe(df_preparado.head())

            # Entrenar el modelo
            try:
                modelo, mae, mse = entrenar_modelo(df_preparado, frecuencia)
                st.session_state.modelo_entrenado = modelo  # Guardar el modelo entrenado en la sesión

                # Mostrar métricas
                st.write(f"MAE: {mae}")
                st.write(f"MSE: {mse}")
                st.success("Modelo entrenado exitosamente.")
            except Exception as e:
                st.error(f"Hubo un error al entrenar el modelo: {e}")


























