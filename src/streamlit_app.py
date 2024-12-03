import streamlit as st
import pandas as pd
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
from model_training import train_model
from data_loading import load_data
from data_preprocessing import preprocess_data

def save_predictions(predictions, dates, productos):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    data = {'fecha': dates, 'producto': productos, 'cantidad_vendida_predicha': predictions, 'modelo': 'RandomForest'}
    for i in range(len(dates)):
        row = {key: data[key][i] for key in data}
        supabase.table('predicciones').insert(row).execute()

# Configurar la página de Streamlit
st.title("Predicción de Stock de Verduras")
st.sidebar.header("Configuraciones")

# Entradas del usuario
producto = st.sidebar.selectbox("Producto", ['tomate', 'lechuga', 'pepino', 'zanahoria', 'cebolla'])
fecha = st.sidebar.date_input("Fecha")
inventario_actual = st.sidebar.number_input("Inventario Actual", min_value=0)

# Entrenar el modelo
if st.sidebar.button("Entrenar Modelo"):
    model = train_model()
    st.write("Modelo entrenado con éxito.")
else:
    model = None

# Hacer predicciones
if model and st.sidebar.button("Hacer Predicciones"):
    df = load_data()
    print("Datos cargados:", df.head())  # Verificar los datos cargados
    df = preprocess_data(df)
    print("Datos preprocesados:", df.head())  # Verificar los datos preprocesados

    X_new = pd.DataFrame({
        'producto': [producto],
        'precio': [df['precio'].mean()],
        'promocion': [0],
        'proveedor': [df['proveedor'].mode()[0]],
        'ubicacion': [df['ubicacion'].mode()[0]],
        'condiciones_climaticas': [df['condiciones_climaticas'].mode()[0]],
        'dia': [fecha.day],
        'mes': [fecha.month],
        'año': [fecha.year],
        'descuento_aplicado': [0],
        'tipo_producto': [df['tipo_producto'].mode()[0]],
        'categoria_producto': [df['categoria_producto'].mode()[0]],
        'hora_venta': [df['hora_venta'].mode()[0]],
        'canal_venta': [df['canal_venta'].mode()[0]],
        'campana_marketing': [0]
    })
    X_new = pd.get_dummies(X_new, columns=['producto', 'proveedor', 'ubicacion', 'condiciones_climaticas'], drop_first=True)
    
    # Asegurarse de que las columnas coincidan
    X_new = X_new.reindex(columns = df.columns, fill_value=0)
    
    prediccion = model.predict(X_new)[0]
    st.write(f"Predicción de ventas para el {fecha}: {prediccion} unidades.")
    
    if inventario_actual < prediccion:
        st.write("Se recomienda comprar más stock.")
    else:
        st.write("No es necesario comprar más stock.")
    
    # Guardar la predicción
    save_predictions([prediccion], [fecha], [producto])

# Mostrar predicciones
if st.sidebar.checkbox("Mostrar Predicciones"):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table('predicciones').select('*').execute()
    df_predicciones = pd.DataFrame(response.data)
    st.write(df_predicciones)

# Mostrar datos originales
if st.sidebar.checkbox("Mostrar Datos Originales"):
    df = load_data()
    st.write(df)

