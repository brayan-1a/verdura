import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from supabase import create_client, Client

# Conectar a Supabase (Agrega tu URL y clave de API)
url = "https://odlosqyzqrggrhvkdovj.supabase.co"  # URL de tu proyecto Supabase
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck"  # Clave pública anon de tu proyecto Supabase
supabase: Client = create_client(url, key)

# Paso 1: Preparar los datos (limpiar y mostrar datos)
def load_and_preprocess_data():
    # Consultar los datos desde Supabase
    response = supabase.table("datos_verduras").select("*").execute()
    
    # Verificar si la consulta fue exitosa
    if response.status_code == 200:
        df = pd.DataFrame(response.data)  # Convertir los datos a un DataFrame
        # Limpieza de datos: eliminar valores nulos o duplicados
        df = df.dropna()  # Eliminar filas con valores nulos
        df = df.drop_duplicates()  # Eliminar filas duplicadas
        
        # Mostrar datos limpios
        st.write("Datos Limpios", df.head())
        return df
    else:
        st.error("Error al obtener los datos desde Supabase.")
        return None

# Paso 2: Entrenamiento del modelo
def train_model(df):
    # Preprocesar los datos: transformar variables categóricas y separar en características y objetivo
    df['producto'] = LabelEncoder().fit_transform(df['producto'])
    df['proveedor'] = LabelEncoder().fit_transform(df['proveedor'])
    df['ubicacion'] = LabelEncoder().fit_transform(df['ubicacion'])
    
    # Seleccionar características y objetivo
    X = df[['producto', 'precio', 'cantidad_vendida', 'promocion', 'coste_adquisicion', 
            'ubicacion', 'inventario_inicial', 'clientes_frecuentes', 'metodo_pago', 
            'ventas_por_hora', 'desperdicio', 'hora_venta', 'canal_venta']]
    
    y = df['inventario_final']  # Predecir inventario final
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Mostrar mensaje de éxito
    st.success("Modelo entrenado con éxito.")
    return model, X_test, y_test

# Paso 3: Evaluación del modelo
def evaluate_model(model, X_test, y_test):
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluación del modelo: calcular error cuadrático medio (RMSE)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Mostrar resultados de la evaluación
    st.write(f"Error Cuadrático Medio (MSE): {mse}")
    st.write(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")

# Paso 4: Predicción
def make_prediction(model):
    # Crear formulario para ingresar datos
    st.subheader("Ingresa los datos para predecir el stock")
    
    # Los datos que se deben ingresar para hacer la predicción
    producto = st.selectbox("Producto", ['tomate', 'lechuga', 'pepino', 'zanahoria', 'cebolla'])
    precio = st.number_input("Precio", min_value=0.1, max_value=10.0, step=0.1)
    cantidad_vendida = st.number_input("Cantidad Vendida", min_value=0, max_value=1000, step=1)
    promocion = st.selectbox("Promoción", [0, 1])
    coste_adquisicion = st.number_input("Costo de Adquisición", min_value=0.0, max_value=10.0, step=0.1)
    ubicacion = st.selectbox("Ubicación", ['Sucursal 1', 'Sucursal 2', 'Sucursal 3'])
    inventario_inicial = st.number_input("Inventario Inicial", min_value=0, max_value=1000, step=1)
    clientes_frecuentes = st.number_input("Clientes Frecuentes", min_value=0, max_value=100, step=1)
    metodo_pago = st.selectbox("Método de Pago", ['Efectivo', 'Tarjeta', 'Transferencia'])
    ventas_por_hora = st.number_input("Ventas por Hora", min_value=0, max_value=50, step=1)
    desperdicio = st.number_input("Desperdicio", min_value=0, max_value=50, step=1)
    hora_venta = st.number_input("Hora de Venta", min_value=0, max_value=23, step=1)
    canal_venta = st.selectbox("Canal de Venta", ['tienda física', 'online'])
    
    # Preparar los datos para la predicción
    input_data = np.array([producto, precio, cantidad_vendida, promocion, coste_adquisicion, 
                           ubicacion, inventario_inicial, clientes_frecuentes, metodo_pago, 
                           ventas_por_hora, desperdicio, hora_venta, canal_venta]).reshape(1, -1)
    
    # Codificar las variables categóricas (como producto, ubicación, etc.)
    input_data[:, 0] = LabelEncoder().fit(['tomate', 'lechuga', 'pepino', 'zanahoria', 'cebolla']).transform([producto])
    input_data[:, 5] = LabelEncoder().fit(['Sucursal 1', 'Sucursal 2', 'Sucursal 3']).transform([ubicacion])
    input_data[:, 8] = LabelEncoder().fit(['Efectivo', 'Tarjeta', 'Transferencia']).transform([metodo_pago])
    input_data[:, 11] = LabelEncoder().fit(['tienda física', 'online']).transform([canal_venta])
    
    # Realizar la predicción
    prediction = model.predict(input_data)
    
    # Mostrar la predicción
    st.write(f"Predicción del inventario final: {prediction[0]}")

# Función principal de Streamlit
def main():
    st.title("Predicción de Stock de Productos")
    
    # Paso 1: Mostrar los datos limpios
    if st.button("Cargar y limpiar datos"):
        df = load_and_preprocess_data()

    # Paso 2: Entrenar el modelo
    if st.button("Entrenar modelo"):
        model, X_test, y_test = train_model(df)

    # Paso 3: Evaluar el modelo
    if st.button("Evaluar modelo"):
        evaluate_model(model, X_test, y_test)

    # Paso 4: Hacer predicción
    if st.button("Hacer predicción"):
        make_prediction(model)

if __name__ == "__main__":
    main()

