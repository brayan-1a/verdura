import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from conexion import obtener_datos
from preparar_datos import preparar_datos_modelo
from modelo import entrenar_y_evaluar, analizar_errores
from sklearn.metrics import mean_squared_error, r2_score

def cargar_modelo(file):
    """Carga un modelo desde un archivo .pkl."""
    try:
        modelo = pickle.load(file)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def main():
    st.title('Evaluación del Modelo Optimizado - Tienda de Verduras')

    # Inicializar estado
    if 'modelo_entrenado' not in st.session_state:
        st.session_state.modelo_entrenado = False
    
    # Tabs para separar funcionalidades
    tab1, tab2 = st.tabs(["Entrenar Modelo", "Cargar Modelo Preentrenado"])
    
    # Tab 1: Entrenar modelo
    with tab1:
        # Cargar datos
        if 'df_ventas' not in st.session_state:
            with st.spinner('Cargando datos...'):
                st.session_state.df_ventas = obtener_datos()
                st.success('Datos cargados correctamente')
        
        # Mostrar muestra de datos
        st.subheader('Muestra de Datos')
        st.dataframe(st.session_state.df_ventas.head())
        
        # Botón para entrenar el modelo
        if st.button('Entrenar Modelo Optimizado', type='primary'):
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
            
            # Visualización de predicciones
            st.subheader('Predicciones vs Valores Reales')
            fig_predictions = px.scatter(
                st.session_state.resultados,
                x='Valor Real',
                y='Predicción',
                title='Comparación de Predicciones vs Valores Reales'
            )
            st.plotly_chart(fig_predictions)
    
    # Tab 2: Cargar modelo preentrenado
    with tab2:
        st.subheader("Cargar Modelo Preentrenado")
        archivo = st.file_uploader("Sube tu modelo entrenado (.pkl)", type=["pkl"])
        
        if archivo is not None:
            modelo = cargar_modelo(archivo)
            
            if modelo is not None:
                st.write("### Modelo Cargado Exitosamente")
                st.write(f"Modelo: {modelo}")  # Información básica del modelo
                
                # Subir datos de prueba
                archivo_datos = st.file_uploader("Sube datos de prueba (CSV)", type=["csv"])
                
                if archivo_datos is not None:
                    datos = pd.read_csv(archivo_datos)
                    st.write("### Datos Cargados")
                    st.dataframe(datos.head())
                    
                    if 'target' in datos.columns:
                        X_test = datos.drop(columns=['target'])
                        y_test = datos['target']
                        
                        # Calcular métricas
                        predicciones = modelo.predict(X_test)
                        rmse = mean_squared_error(y_test, predicciones, squared=False)
                        r2 = r2_score(y_test, predicciones)
                        
                        st.write("### Resultados del Modelo")
                        st.write(f"RMSE: {rmse:.2f}")
                        st.write(f"R²: {r2:.2f}")
                        
                        # Mostrar predicciones
                        datos['Predicciones'] = predicciones
                        st.write("### Predicciones vs Valores Reales")
                        st.dataframe(datos[['target', 'Predicciones']])
                    else:
                        st.error("El archivo de datos debe contener una columna llamada 'target'.")
            else:
                st.error("Hubo un error al cargar el modelo. Intenta nuevamente.")

if __name__ == '__main__':
    main()





































