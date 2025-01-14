import streamlit as st
from supabase import create_client

# Conexión a Supabase
url = "https://odlosqyzqrggrhvkdovj.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck"

def main():
    st.title("Test de conexión Supabase")
    
    try:
        # Crear cliente
        supabase = create_client(url, key)
        
        # Listar todas las tablas
        response = supabase.table('ventas').select("*").execute()
        
        # Mostrar resultado
        st.write("Respuesta de Supabase:", response)
        
    except Exception as e:
        st.error(f"Error de conexión: {str(e)}")

if __name__ == "__main__":
    main()



