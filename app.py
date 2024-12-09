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

# Limpiar los datos
df_clean = clean_data(df)
st.write("Datos Limpiados:", df_clean)

# Normalizar columnas numéricas
numeric_columns = ["precio", "cantidad_vendida", "inventario_inicial", "inventario_final", "desperdicio"]
df_norm = normalize_data(df_clean, numeric_columns)
st.write("Datos Normalizados:", df_norm)


# Mostrar los datos seleccionados en Streamlit
st.title("Análisis de Predicción de Stock - Verduras")
st.write("Datos seleccionados:", df)

