import streamlit as st
import pandas as pd
from conexion import obtener_datos
from preparar_datos import preparar_datos_modelo
from modelo import entrenar_modelo, predecir_stock

def main():
    st.title('Predicción de Stock - Tienda de Verduras')
    
    # Cargar datos
    with st.spinner('Cargando datos...'):
        df_ventas, df_inventarios, df_desperdicios = obtener_datos()
        
    # Preparar datos
    df_final = preparar_datos_modelo(df_ventas, df_inventarios, df_desperdicios)
    
    # Entrenar modelo
    with st.spinner('Entrenando modelo...'):
        modelo, rmse, r2 = entrenar_modelo(df_final)
        
    # Mostrar métricas
    st.subheader('Métricas del Modelo')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('RMSE', round(rmse, 2))
    with col2:
        st.metric('R²', round(r2, 2))
    
    # Sección de predicción
    st.subheader('Realizar Predicción')
    producto_id = st.selectbox('Seleccionar Producto', df_ventas['producto_id'].unique())
    dia_semana = st.selectbox('Día de la Semana', range(7))
    mes = st.selectbox('Mes', range(1, 13))
    
    if st.button('Predecir'):
        datos_nuevos = pd.DataFrame({
            'dia_semana': [dia_semana],
            'mes': [mes],
            'cantidad_perdida': [df_final['cantidad_perdida'].mean()]  # Usamos el promedio histórico
        })
        prediccion = predecir_stock(modelo, datos_nuevos)
        st.success(f'Stock recomendado: {round(prediccion[0])} unidades')

if __name__ == '__main__':
    main()





























