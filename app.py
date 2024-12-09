import streamlit as st
from config import get_supabase_client
from preprocess import load_and_select_data

# Conexión con Supabase
supabase = get_supabase_client()

# Columnas relevantes
selected_columns = [
    "fecha",
    "producto",
    "precio",
    "cantidad_vendida",
    "promocion",
    "inventario_inicial",
    "inventario_final",
    "desperdicio",
    "condiciones_climaticas",
    "ventas_por_hora"
]

# Cargar los datos
df = load_and_select_data(supabase, "verduras", selected_columns)

# Mostrar los datos seleccionados en Streamlit
st.title("Análisis de Predicción de Stock - Verduras")
st.write("Datos seleccionados:", df)

