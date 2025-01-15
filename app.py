import streamlit as st
import pandas as pd
import plotly.express as px
from modelo import entrenar_y_evaluar, analizar_errores
from preparar_datos import preparar_datos_modelo

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Stock - Tienda de Verduras",
    page_icon="🥬",
    layout="wide"
)

def main():
    st.title('🥬 Predicción de Stock - Tienda de Verduras')

    if 'modelo_entrenado' not in st.session_state:
        st.session_state.modelo_entrenado = False

    if 'df_ventas' not in st.session_state:
        # Aquí se cargarían los datos como en el código anterior
        pass

    # Mostrar muestra de datos
    st.subheader('📊 Muestra de Datos')
    st.dataframe(st.session_state.df_ventas.head())

    # Botón para entrenar el modelo
    if st.button('🚀 Entrenar Modelo de Stock', type='primary'):
        try:
            st.session_state.modelo_entrenado = True
            df_preparado = preparar_datos_modelo(st.session_state.df_ventas)
            modelo, resultados, metricas, importancia = entrenar_y_evaluar(df_preparado)
            
            # Guardamos el modelo entrenado en el estado de sesión
            st.session_state.modelo = modelo
            st.session_state.resultados = resultados
            st.session_state.metricas = metricas
            st.session_state.importancia = importancia
            
            st.success('✨ ¡Modelo entrenado exitosamente!')

        except Exception as e:
            st.error(f'❌ Error durante el entrenamiento: {str(e)}')
            return

    # Mostrar resultados si el modelo ha sido entrenado
    if st.session_state.modelo_entrenado:
        # Mostrar métricas, importancia de características y análisis de errores

        # Nueva pestaña para selección de producto
        st.subheader('🎯 Recomendación de Stock')
        
        # Selección del producto
        productos = st.session_state.df_ventas['nombre_producto'].unique()
        producto_seleccionado = st.selectbox('Selecciona el producto', productos)

        if st.button('🔮 Predecir Stock para el Producto'):
            # Filtrar el producto seleccionado
            producto_data = st.session_state.df_ventas[st.session_state.df_ventas['nombre_producto'] == producto_seleccionado].iloc[0]
            
            # Extraer las características del producto seleccionado
            caracteristicas_producto = producto_data[['ventas_7d', 'variabilidad_ventas', 'tasa_perdida', 
                                                      'dia_semana', 'mes', 'es_fin_semana']].values.reshape(1, -1)
            
            # Normalizar las características del producto
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            caracteristicas_normalizadas = scaler.fit_transform(caracteristicas_producto)
            
            # Hacer la predicción
            modelo = st.session_state.modelo
            stock_predicho = modelo.predict(caracteristicas_normalizadas)[0]
            
            # Mostrar la recomendación
            st.write(f"🔮 La recomendación de stock para el producto '{producto_seleccionado}' es: {stock_predicho:.2f} unidades.")

if __name__ == '__main__':
    main()










































