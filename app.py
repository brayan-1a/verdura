import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Stock - Tienda de Verduras",
    page_icon="🥬",
    layout="wide"
)

# Personalización de estilo con CSS
st.markdown("""
    <style>
        /* Fondo de la página */
        .reportview-container {
            background-color: #F4F4F9;
        }

        /* Barra lateral */
        .sidebar .sidebar-content {
            background-color: #2E8B57;
            color: #FFFFFF;
        }

        /* Títulos de la aplicación */
        .css-18e3th9 {
            color: #2C6B2F;
        }

        /* Color de los botones */
        .stButton>button {
            background-color: #2E8B57;
            color: white;
            border-radius: 5px;
        }

        /* Cuando pasas el mouse sobre los botones */
        .stButton>button:hover {
            background-color: #FFCC00;
        }

        /* Color de las métricas */
        .stMetric>div {
            color: #2C6B2F;
        }

        /* Color de las tablas */
        .stDataFrame {
            color: #333333;
        }

        /* Títulos de las secciones */
        .css-10trblm {
            color: #2C6B2F;
        }

        /* Gráficos */
        .plotly-graph-div {
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Importar funciones locales
from conexion import obtener_datos
from preparar_datos import preparar_datos_modelo
from modelo import entrenar_y_evaluar, analizar_errores

# Diccionario de productos y sus IDs
productos_dict = {
    'Tomate': 1,
    'Pepino': 2,
    'Zanahoria': 3,
    'Lechuga': 4,
    'Cebolla': 5
}

def main():
    st.title('🥬 Predicción de Stock - Tienda de Verduras')

    # Barra lateral para elegir entre las secciones
    menu = st.sidebar.radio("Selecciona una opción", ["Entrenar Modelo", "Predicción de Stock", "Gráficos"])

    # Inicializar estado
    if 'modelo_entrenado' not in st.session_state:
        st.session_state.modelo_entrenado = False
    
    # Cargar datos
    if 'df_ventas' not in st.session_state:
        with st.spinner('Cargando datos de Supabase...'):
            try:
                st.session_state.df_ventas = obtener_datos()
                if not st.session_state.df_ventas.empty:
                    st.success('✅ Datos cargados correctamente')
                else:
                    st.warning('⚠️ No se encontraron datos en la base de datos')
                    return
            except Exception as e:
                st.error(f'❌ Error al cargar datos: {str(e)}')
                st.info('📌 Verifica la conexión con Supabase y los datos disponibles')
                return

    # Entrenamiento del Modelo
    if menu == "Entrenar Modelo":
        # Botón para entrenar el modelo
        if st.button('🚀 Entrenar Modelo de Stock', type='primary'):
            try:
                st.session_state.modelo_entrenado = True
                
                with st.spinner('🔄 Preparando datos...'):
                    df_preparado = preparar_datos_modelo(st.session_state.df_ventas)
                    st.success('✅ Datos preparados correctamente')
                
                # Entrenar modelo y obtener resultados
                with st.spinner('⚙️ Entrenando modelo...'):
                    modelo, resultados, metricas, importancia = entrenar_y_evaluar(df_preparado)
                    error_analysis = analizar_errores(resultados)
                    
                    st.session_state.resultados = resultados
                    st.session_state.metricas = metricas
                    st.session_state.importancia = importancia
                    st.session_state.error_analysis = error_analysis
                    st.session_state.modelo = modelo  # Guardamos el modelo en session_state
                    
                    st.success('✨ ¡Modelo entrenado exitosamente!')
            except Exception as e:
                st.error(f'❌ Error durante el entrenamiento: {str(e)}')
                return

        # Mostrar resultados si el modelo ha sido entrenado
        if st.session_state.modelo_entrenado:
            try:
                # Métricas principales
                st.subheader('📈 Métricas del Modelo')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('R² (Test)', f"{st.session_state.metricas['r2_test']:.3f}")
                with col2:
                    st.metric('RMSE (Test)', f"{st.session_state.metricas['rmse_test']:.2f}")
                with col3:
                    st.metric('R² CV Promedio', f"{st.session_state.metricas['cv_scores_mean']:.3f}")
                
                # Importancia de características
                st.subheader('🎯 Importancia de Características')
                fig_importance = px.bar(
                    st.session_state.importancia,
                    x='caracteristica',
                    y='importancia',
                    title='Importancia de cada característica en el modelo'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Análisis de errores
                st.subheader('📊 Análisis de Stock')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Error Medio (unidades)', 
                             f"{st.session_state.error_analysis['error_medio_unidades']:.2f}")
                    st.metric('Error Máximo (unidades)', 
                             f"{st.session_state.error_analysis['maximo_error_unidades']:.2f}")
                with col2:
                    st.metric('Stock Insuficiente (%)', 
                             f"{st.session_state.error_analysis['stock_insuficiente']:.1f}%")
                    st.metric('Error Mediano (unidades)', 
                             f"{st.session_state.error_analysis['error_mediano_unidades']:.2f}")
                
                # Visualización de predicciones
                st.subheader('🎯 Predicciones vs Valores Reales')
                fig_predictions = px.scatter(
                    st.session_state.resultados,
                    x='Stock Real',
                    y='Stock Predicho',
                    title='Comparación de Stock Predicho vs Real',
                    labels={'Stock Real': 'Stock Real (unidades)', 
                           'Stock Predicho': 'Stock Predicho (unidades)'}
                )
                fig_predictions.add_shape(
                    type='line', line=dict(dash='dash'),
                    x0=st.session_state.resultados['Stock Real'].min(),
                    y0=st.session_state.resultados['Stock Real'].min(),
                    x1=st.session_state.resultados['Stock Real'].max(),
                    y1=st.session_state.resultados['Stock Real'].max()
                )
                st.plotly_chart(fig_predictions, use_container_width=True)
                
            except Exception as e:
                st.error(f'❌ Error al mostrar resultados: {str(e)}')

    # Predicción de Stock
    elif menu == "Predicción de Stock":
        if st.session_state.modelo_entrenado:
            # Permitir seleccionar un producto
            producto_seleccionado = st.selectbox(
                "Selecciona un producto",
                ["Tomate", "Pepino", "Zanahoria", "Lechuga", "Cebolla"]
            )
            producto_id = productos_dict[producto_seleccionado]

            # Obtener el DataFrame de las ventas del producto seleccionado
            df_producto = st.session_state.df_ventas[st.session_state.df_ventas['producto_id'] == producto_id]

            if df_producto.empty:
                st.warning("No se encontraron datos para este producto.")
            else:
                # Mostrar información sobre el producto
                st.subheader(f"Predicción de Stock para el Producto: {producto_seleccionado}")
                
                # Botón para hacer la predicción
                if st.button('📦 Predecir Stock'):
                    try:
                        # Preparar datos para la predicción usando la función actualizada
                        df_preparado = preparar_datos_modelo(df_producto)
                        
                        # Verificar si el modelo está en session_state
                        if 'modelo' not in st.session_state:
                            st.error("❌ El modelo no está disponible. Por favor, entrene el modelo primero.")
                            return

                        # Realizar la predicción usando todas las características necesarias
                        modelo = st.session_state.modelo
                        X = df_preparado[[ 
                            'ventas_7d',
                            'variabilidad_ventas',
                            'tasa_perdida',
                            'dia_semana',
                            'mes',
                            'es_fin_semana',
                            'ventas_14d',
                            'ventas_fin_semana',
                            'stock_medio',
                            'tasa_rotacion',
                            'tendencia_ventas'
                        ]].iloc[-1:]  # Tomar solo la última fila para la predicción

                        prediccion = modelo.predict(X)

                        # Mostrar la predicción y datos relevantes
                        st.subheader('💡 Recomendación de Stock')
                        st.write(f"Recomendación de Stock para el próximo periodo: {prediccion[0]:.0f} unidades")
                        
                        # Mostrar métricas adicionales que ayuden a entender la predicción
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Ventas promedio (7 días)", 
                                    f"{X['ventas_7d'].iloc[0]:.1f}")
                            st.metric("Tasa de rotación", 
                                    f"{X['tasa_rotacion'].iloc[0]:.2f}")
                        with col2:
                            st.metric("Tendencia de ventas", 
                                    f"{X['tendencia_ventas'].iloc[0]:.1%}")
                            st.metric("Tasa de pérdida", 
                                    f"{X['tasa_perdida'].iloc[0]:.1%}")

                    except Exception as e:
                        st.error(f'❌ Error al predecir el stock: {str(e)}')
                        st.info('📌 Detalles del error para debugging: ' + str(e))

        else:
            st.warning("⚠️ No se ha entrenado el modelo aún. Entrénalo primero.")

    # Gráficos
    elif menu == "Gráficos":
        st.subheader("📊 Gráficos de Ventas e Inventarios")
        
        # Gráfico de ventas por producto
        st.subheader("Ventas Totales por Producto")
        ventas_por_producto = st.session_state.df_ventas.groupby("producto_id")["cantidad_vendida"].sum().reset_index()
        fig_ventas = px.bar(
            ventas_por_producto,
            x="producto_id",
            y="cantidad_vendida",
            labels={"producto_id": "Producto", "cantidad_vendida": "Ventas Totales"},
            title="Ventas Totales por Producto"
        )
        st.plotly_chart(fig_ventas, use_container_width=True)
        
        # Gráfico de inventarios por producto
        st.subheader("Inventarios a lo largo del tiempo")
        inventarios_por_producto = st.session_state.df_ventas.groupby(["producto_id", "fecha_venta"])["inventario_final"].sum().reset_index()
        fig_inventarios = px.line(
            inventarios_por_producto,
            x="fecha_venta",
            y="inventario_final",
            color="producto_id",
            labels={"fecha_venta": "Fecha", "inventario_final": "Inventario Final"},
            title="Evolución del Inventario por Producto"
        )
        st.plotly_chart(fig_inventarios, use_container_width=True)

if __name__ == '__main__':
    main()














































