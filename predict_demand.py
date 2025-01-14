import streamlit as st
from supabase import create_client
import pandas as pd

def main():
    st.title("Análisis detallado de datos Supabase")
    
    try:
        # Conexión a Supabase
        url = 'https://odlosqyzqrggrhvkdovj.supabase.co'
        key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck'
        supabase = create_client(url, key)
        
        # Obtener datos de la tabla ventas
        response = supabase.table('ventas').select("*").execute()
        
        # Mostrar datos crudos
        st.subheader("Datos crudos de la respuesta:")
        st.json(response)
        
        # Intentar convertir a DataFrame
        if 'data' in response:
            df = pd.DataFrame(response['data'])
            st.subheader("Estructura del DataFrame:")
            st.write("Columnas:", list(df.columns))
            st.write("Primeras filas:")
            st.write(df.head())
            st.write("Información del DataFrame:")
            st.write(df.info())
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()



