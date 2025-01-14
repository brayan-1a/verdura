import streamlit as st
import pandas as pd
from conexion import obtener_datos
from preparar_datos import preparar_datos_modelo
from modelo import entrenar_y_evaluar

def main():
    st.title('Evaluación del Modelo - Tienda de Verduras')
    
    # Inicializar variables de estado en session_state
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
    
    # Información sobre los datos
    st.subheader('Información del Dataset')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Registros', len(st.session_state.df_ventas))
    with col2:
        st.metric('Productos Únicos', st.session_state.df_ventas['producto_id'].nunique())
    with col3:
        st.metric('Rango de Fechas', f"{st.session_state.df_ventas['fecha_venta'].min()} a {st.session_state.df_ventas['fecha_venta'].max()}")
    
    # Botón para entrenar el modelo
    if st.button('Entrenar Modelo', type='primary'):
        st.session_state.modelo_entrenado = True
        
        with st.spinner('Preparando datos...'):
            df_preparado = preparar_datos_modelo(st.session_state.df_ventas)
            st.success('Datos preparados correctamente')
        
        # Entrenar modelo y obtener resultados
        with st.spinner('Entrenando modelo...'):
            modelo, resultados = entrenar_y_evaluar(df_preparado)
            st.session_state.resultados = resultados
            
            # Calcular métricas
            rmse = resultados['Diferencia'].mean()
            r2 = 1 - (resultados['Diferencia'].sum() ** 2) / ((resultados['Valor Real'] - resultados['Valor Real'].mean()) ** 2).sum()
            
            st.session_state.rmse = rmse
            st.session_state.r2 = r2
            
            st.success('¡Modelo entrenado exitosamente!')
    
    # Mostrar resultados solo si el modelo ha sido entrenado
    if st.session_state.modelo_entrenado:
        st.subheader('Métricas del Modelo')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('RMSE (Error Medio)', round(st.session_state.rmse, 2))
            st.caption('Menor RMSE indica mejor precisión')
        with col2:
            st.metric('R² (Coeficiente de Determinación)', round(st.session_state.r2, 2))
            st.caption('R² más cercano a 1 indica mejor ajuste')
        
        # Explicación de las métricas
        with st.expander("📊 Interpretación de las Métricas"):
            st.write("""
            - **RMSE (Root Mean Square Error)**: Mide el error promedio de las predicciones. 
              Un valor más bajo indica predicciones más precisas.
            
            - **R² (R-cuadrado)**: Indica qué tan bien el modelo explica la variabilidad de los datos.
              - R² = 1: ajuste perfecto
              - R² = 0: el modelo no explica la variabilidad
              - R² negativo: el modelo necesita mejoras
            """)
        
        # Mostrar resultados detallados
        st.subheader('Comparación de Resultados')
        st.dataframe(st.session_state.resultados.head(10))
        
        # Visualización
        st.subheader('Visualización de Predicciones vs Valores Reales')
        chart_data = st.session_state.resultados[['Valor Real', 'Predicción']].head(20)
        st.line_chart(chart_data)

if __name__ == '__main__':
    main()





























