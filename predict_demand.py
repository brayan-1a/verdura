import streamlit as st
from supabase import create_client
import pandas as pd

# Conexión a Supabase
url = "https://odlosqyzqrggrhvkdovj.supabase.co"  # URL de Supabase
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck"  # Tu clave de Supabase

# Crear cliente Supabase
supabase = create_client(url, key)

# Función para cargar datos desde la tabla 'ventas'
def cargar_datos():
    try:
        # Realizar la consulta en la tabla ventas
        response = supabase.table('ventas').select("*").execute()

        if response.status_code == 200:
            st.write("Datos cargados correctamente.")
            return pd.DataFrame(response.data)  # Devolver como DataFrame
        else:
            st.error("Error al cargar los datos. Verifica la respuesta de Supabase.")
            return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error
    except Exception as e:
        st.error(f"Error en la conexión: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("Visualización de Datos de Ventas desde Supabase")
    
    # Botón para cargar los datos
    if st.button("Cargar Datos de Ventas"):
        df = cargar_datos()
        if not df.empty:
            st.write("Primeras filas de los datos:")
            st.write(df.head())  # Muestra las primeras filas del DataFrame

if __name__ == "__main__":
    main()




