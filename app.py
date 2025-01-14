import streamlit as st
import pandas as pd
from conexion import obtener_datos
from preparar_datos import preparar_datos_modelo
from modelo import entrenar_y_evaluar

def main():
    st.title('Evaluación del Modelo - Tienda de Verduras')
    
    # Cargar datos
    with st.spinner('Cargando datos...'):
        df_ventas = obtener_datos()
        st.success('Datos cargados correctamente')
        
    # Mostrar muestra de datos
    st.subheader('Muestra de Datos')
    st.dataframe(df_ventas.head())
        
    # Preparar datos
    df_preparado = preparar_datos_modelo(df_ventas)
    
    # Entrenar modelo y obtener resultados
    with st.spinner('Entrenando modelo...'):
        modelo, resultados = entrenar_y_evaluar(df_preparado)
        
        # Obtener métricas del modelo
        rmse = resultados['Diferencia'].mean()
        r2 = 1 - (resultados['Diferencia'].sum() ** 2) / ((resultados['Valor Real'] - resultados['Valor Real'].mean()) ** 2).sum()
        
    # Mostrar métricas
    st.subheader('Métricas del Modelo')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('RMSE', round(rmse, 2))
    with col2:
        st.metric('R²', round(r2, 2))
    
    # Mostrar resultados del modelo
    st.subheader('Comparación de Resultados')
    st.dataframe(resultados.head(10))
    
    # Mostrar gráfico de comparación
    st.subheader('Visualización de Resultados')
    st.line_chart(resultados[['Valor Real', 'Predicción']].head(20))

if __name__ == '__main__':
    main()





























