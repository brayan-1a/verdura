import streamlit as st
import pandas as pd
import plotly.express as px
from conexion import obtener_datos
from preparar_datos import preparar_datos_modelo
from modelo import entrenar_y_evaluar, analizar_errores

def main():
    st.title('Evaluación del Modelo - Predicción de Stock Necesario')
    
    # Inicializar estado
    if 'modelo_entrenado' not in st.session_state:
        st.session_state.modelo_entrenado = False
    
    # Cargar datos
    if 'df_ventas' not in st.session_state:
        with st.spinner('Cargando datos...'):
            st.session_state.df_ventas = obtener_datos()
            st.success('Datos cargados correctamente')
    
    # Mostrar muestra de datos
    st.subheader('Muestra de Datos')
    st.dataframe(st.session_state.df_ventas.head())
    
    # Botón para entrenar el modelo
    if st.button('Entrenar Modelo para Predicción de Stock', type='primary'):
        st.session_state.modelo_entrenado = True
        
        with st.spinner('Preparando datos...'):
            df_preparado = preparar_datos_modelo(st.session_state.df_ventas)
            st.success('Datos preparados correctamente')
        
        # Entrenar modelo y obtener resultados
        with st.spinner('Entrenando modelo...'):
            modelo, resultados, metricas, importancia = entrenar_y_evaluar(df_preparado)
            error_analysis = analizar_errores(resultados)
            
            st.session_state.resultados = resultados
            st.session_state.metricas = metricas
            st.session_state.importancia = importancia
            st.session_state.error_analysis = error_analysis
            
            st.success('¡Modelo entrenado exitosamente!')
    
    # Mostrar resultados si el modelo ha sido entrenado
    if st.session_state.modelo_entrenado:
        # Métricas principales
        st.subheader('Métricas del Modelo')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('R² (Test)', f"{st.session_state.metricas['r2_test']:.3f}")
        with col2:
            st.metric('RMSE (Test)', f"{st.session_state.metricas['rmse_test']:.2f}")
        with col3:
            st.metric('R² CV Promedio', f"{st.session_state.metricas['cv_scores_mean']:.3f}")
        
        # Importancia de características
        st.subheader('Importancia de Características')
        fig_importance = px.bar(
            st.session_state.importancia,
            x='caracteristica',
            y='importancia',
            title='Importancia de cada característica en el modelo'
        )
        st.plotly_chart(fig_importance)
        
        # Análisis de errores
        st.subheader('Análisis de Errores')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Error Medio', f"{st.session_state.error_analysis['error_medio']:.2f}")
            st.metric('Error Máximo', f"{st.session_state.error_analysis['error_max']:.2f}")
        with col2:
            st.metric('Error Mediano', f"{st.session_state.error_analysis['error_mediano']:.2f}")
            st.metric('Desviación Estándar', f"{st.session_state.error_analysis['error_std']:.2f}")
        
        # Visualización de predicciones de stock necesario
        st.subheader('Predicción de Stock Necesario')
        fig_predictions = px.scatter(
            st.session_state.resultados,
            x='Fecha',
            y='Predicción de Stock Necesario',
            title='Predicción de Stock Necesario vs. Fecha'
        )
        st.plotly_chart(fig_predictions)

if __name__ == '__main__':
    main()





























