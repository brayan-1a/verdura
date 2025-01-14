import streamlit as st
from preparar_datos import obtener_datos, preparar_datos
from entrenar_modelo import entrenar_modelo
from evaluar_modelo import predecir_demanda
import pandas as pd

st.title("Predicción de Demanda de Productos")

# Selección de período
periodo = st.selectbox("Seleccione el período para la predicción", ["Día", "Semana", "Mes"])
frecuencia = {'Día': 'D', 'Semana': 'W', 'Mes': 'M'}[periodo]

if st.button("Cargar datos y entrenar modelo"):
    # Obtener y preparar datos
    df_ventas = obtener_datos()
    df_preparado = preparar_datos(df_ventas, frecuencia)
    
    # Entrenar modelo
    modelo, mae, mse = entrenar_modelo(df_preparado, frecuencia)
    
    # Mostrar métricas
    st.write(f"MAE: {mae}")
    st.write(f"MSE: {mse}")
    
    # Predecir para el próximo período
    ultimo_periodo = df_preparado['periodo'].max()
    fechas_futuras = pd.date_range(ultimo_periodo, periods=10, freq=frecuencia)
    predicciones = predecir_demanda(modelo, fechas_futuras)
    
    # Mostrar predicciones
    resultados = pd.DataFrame({"Fecha": fechas_futuras, "Demanda Predicha": predicciones})
    st.write("Predicciones para los próximos períodos:")
    st.dataframe(resultados)



















