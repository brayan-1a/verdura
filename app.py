import streamlit as st
from modelo import entrenar_modelo, predecir_stock
from supabase_connector import cargar_productos
from visualizacion import mostrar_grafico_ventas, mostrar_grafico_predicciones
from utils import limpiar_datos, dividir_datos

# Título de la aplicación
st.title("Predicción de Stock para Verduras")

# Cargar productos desde Supabase
df_productos = cargar_productos()

if df_productos.empty:
    st.error("No se pudieron cargar los productos desde la base de datos.")
else:
    # Si los datos se cargaron correctamente, continuar con el flujo
    df_productos_limpios = limpiar_datos(df_productos)
    df_entrenamiento, df_prueba = dividir_datos(df_productos_limpios)

    # Pestañas
    tabs = st.sidebar.radio("Selecciona una opción", ["Entrenamiento de Modelos", "Predicción de Stock", "Visualización"])

    # Entrenamiento de Modelos
    if tabs == "Entrenamiento de Modelos":
        st.subheader("Entrenamiento de Modelos")
        modelo = entrenar_modelo(df_entrenamiento)
        st.write("Modelo entrenado exitosamente")

    # Predicción de Stock
    if tabs == "Predicción de Stock":
        st.subheader("Predicción de Stock")
        precio_unitario = st.number_input("Precio Unitario", min_value=0.0, value=10.0)
        cantidad_promocion = st.number_input("Cantidad en Promoción", min_value=0, value=0)
        temperatura = st.number_input("Temperatura (°C)", value=25.0)
        humedad = st.number_input("Humedad (%)", min_value=0.0, max_value=100.0, value=60.0)

        if st.button("Predecir Stock"):
            cantidad_predicha = predecir_stock(precio_unitario, cantidad_promocion, temperatura, humedad)
            st.write(f"La cantidad recomendada de stock es: {cantidad_predicha}")

    # Visualización de Datos
    if tabs == "Visualización":
        st.subheader("Visualización de Datos")
        mostrar_grafico_ventas(df_productos)
        mostrar_grafico_predicciones(df_productos)



















