import streamlit as st
from config import get_supabase_client
from preprocess import load_and_select_data, clean_data, normalize_data

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
st.title("Análisis de Predicción de Stock - Verduras")
try:
    df = load_and_select_data(supabase, "verduras", selected_columns)
    st.write("Datos seleccionados:", df.head())
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")

# Limpiar los datos
try:
    df_clean = clean_data(df)
    st.write("Datos Limpiados:", df_clean.head())
except Exception as e:
    st.error(f"Error durante la limpieza de datos: {e}")

# Normalizar columnas numéricas
numeric_columns = ["precio", "cantidad_vendida", "inventario_inicial", "inventario_final", "desperdicio"]
try:
    df_norm = normalize_data(df_clean, numeric_columns)
    st.write("Datos Normalizados:", df_norm.head())
except Exception as e:
    st.error(f"Error durante la normalización: {e}")


