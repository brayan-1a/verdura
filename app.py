import streamlit as st
from preparar_datos import obtener_datos, preparar_datos
from entrenar_modelo import entrenar_modelo
from evaluar_modelo import predecir_demanda
import pandas as pd

st.title("Predicción de Demanda de Productos")

# Variable de sesión para guardar el modelo entrenado
if 'modelo_entrenado' not in st.session_state:
    st.session_state.modelo_entrenado = None

# Pestañas para navegar entre "Entrenamiento" y "Predicción"
opcion = st.radio("Selecciona una opción", ("Entrenar Modelo", "Realizar Predicciones"))

if opcion == "Entrenar Modelo":
    # Sección de entrenamiento del modelo
    st.subheader("Entrenamiento del Modelo")
    
    periodo = st.selectbox("Seleccione el período para la predicción", ["Día", "Semana", "Mes"])
    frecuencia = {'Día': 'D', 'Semana': 'W', 'Mes': 'M'}[periodo]

    if st.button("Cargar datos y entrenar modelo"):
        # Obtener y preparar datos
        df_ventas = obtener_datos()
        df_preparado = preparar_datos(df_ventas, frecuencia)

        if df_preparado.empty:
            st.error("No se encontraron datos suficientes para entrenar el modelo.")
        else:
            # Entrenar modelo
            modelo, mae, mse = entrenar_modelo(df_preparado, frecuencia)
            st.session_state.modelo_entrenado = modelo  # Guardar el modelo entrenado en la sesión
            
            # Mostrar métricas
            st.write(f"MAE: {mae}")
            st.write(f"MSE: {mse}")
            st.success("Modelo entrenado exitosamente. Ahora puedes hacer predicciones.")
    
elif opcion == "Realizar Predicciones":
    # Sección de predicción
    if st.session_state.modelo_entrenado:
        st.subheader("Realizar Predicciones")

        # Selección de producto para predecir
        productos = ['Tomate', 'Lechuga', 'Pepino']  # Ejemplo de productos
        producto_seleccionado = st.selectbox("Seleccione el producto", productos)
        
        # Selección de periodo para predicción
        periodo_prediccion = st.selectbox("Seleccione el período de predicción", ["Día", "Semana", "Mes"])
        frecuencia_prediccion = {'Día': 'D', 'Semana': 'W', 'Mes': 'M'}[periodo_prediccion]
        
        if st.button("Predecir demanda"):
            # Obtener la fecha de la última venta
            df_ventas = obtener_datos()
            df_preparado = preparar_datos(df_ventas, frecuencia_prediccion)
            
            # Predecir para los próximos períodos
            if df_preparado.empty:
                st.error("No se encontraron datos suficientes para realizar predicciones.")
            else:
                ultimo_periodo = df_preparado['periodo'].max()
                fechas_futuras = pd.date_range(ultimo_periodo, periods=10, freq=frecuencia_prediccion)
                predicciones = predecir_demanda(st.session_state.modelo_entrenado, fechas_futuras)
                
                # Mostrar resultados
                resultados = pd.DataFrame({"Fecha": fechas_futuras, "Demanda Predicha": predicciones})
                st.write(f"Predicción de demanda para {producto_seleccionado}:")
                st.dataframe(resultados)
    else:
        st.info("Primero entrena el modelo para poder hacer predicciones.")






















