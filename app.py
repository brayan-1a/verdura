import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Conexión a Supabase
url = st.secrets["supabase_url"]  # URL de Supabase
key = st.secrets["supabase_key"]  # Key de Supabase
supabase: Client = create_client(url, key)

# Cargar datos de Supabase
def cargar_datos():
    # Obtener ventas
    ventas_query = supabase.table("ventas").select("*").execute()
    ventas_df = pd.DataFrame(ventas_query['data'])

    # Obtener productos
    productos_query = supabase.table("productos").select("*").execute()
    productos_df = pd.DataFrame(productos_query['data'])

    # Obtener desperdicio
    desperdicio_query = supabase.table("desperdicio").select("*").execute()
    desperdicio_df = pd.DataFrame(desperdicio_query['data'])

    # Obtener promociones
    promociones_query = supabase.table("promociones").select("*").execute()
    promociones_df = pd.DataFrame(promociones_query['data'])

    # Obtener condiciones climáticas
    clima_query = supabase.table("condiciones_climaticas").select("*").execute()
    clima_df = pd.DataFrame(clima_query['data'])

    # Combinar los datos en un único DataFrame
    df = pd.merge(ventas_df, productos_df, on="producto_id", how="left")
    df = pd.merge(df, desperdicio_df, on="producto_id", how="left")
    df = pd.merge(df, promociones_df, on="producto_id", how="left")
    df = pd.merge(df, clima_df, left_on="fecha_venta", right_on="fecha", how="left")

    # Limpiar los datos
    df['cantidad_perdida'].fillna(0, inplace=True)  # Rellenar pérdidas con 0 si no hay datos
    df['descuento_aplicado'].fillna(0, inplace=True)
    df['temperatura'].fillna(df['temperatura'].mean(), inplace=True)
    df['humedad'].fillna(df['humedad'].mean(), inplace=True)
    
    return df

# Preprocesamiento y preparación de datos
def preparar_datos(df):
    # Convertir fechas
    df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
    
    # Variables predictoras (features)
    X = df[['cantidad_vendida', 'descuento_aplicado', 'temperatura', 'humedad', 'cantidad_perdida']]
    
    # Variable objetivo (target): la cantidad de productos vendidos
    y = df['cantidad_vendida']
    
    return X, y

# Entrenar múltiples modelos
def entrenar_modelos(X_train, y_train, X_test, y_test):
    modelos = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        resultados[nombre] = {
            "Modelo": nombre,
            "MSE": mse,
            "R2": r2
        }
    
    return resultados

# Guardar el modelo entrenado
def guardar_modelo(modelo, nombre):
    joblib.dump(modelo, f'{nombre}_modelo.pkl')
    
# Interfaz de Streamlit
def main():
    st.title("Predicción de Stock de Verduras")
    
    # Botón para cargar y entrenar el modelo
    if st.button("Entrenar Modelo"):
        st.write("Cargando datos desde Supabase...")
        df = cargar_datos()
        st.write("Datos cargados correctamente.")
        
        X, y = preparar_datos(df)
        
        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar los modelos
        resultados = entrenar_modelos(X_train, y_train, X_test, y_test)
        
        # Mostrar los resultados
        st.write("Resultados de los modelos:")
        resultados_df = pd.DataFrame(resultados).T
        st.write(resultados_df)
        
        # Selección del mejor modelo según R2
        mejor_modelo_nombre = resultados_df.loc[resultados_df['R2'].idxmax()]["Modelo"]
        st.write(f"El mejor modelo es: {mejor_modelo_nombre}")
        
        # Guardar el mejor modelo
        modelo_seleccionado = next(modelo for nombre, modelo in {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
        }.items() if nombre == mejor_modelo_nombre)
        
        modelo_seleccionado.fit(X_train, y_train)
        
        # Botón para descargar el modelo
        if st.button("Descargar Modelo"):
            guardar_modelo(modelo_seleccionado, mejor_modelo_nombre)
            st.write(f"Modelo {mejor_modelo_nombre} guardado como '{mejor_modelo_nombre}_modelo.pkl'.")
            st.download_button("Haz clic para descargar el modelo", f'{mejor_modelo_nombre}_modelo.pkl')

if __name__ == "__main__":
    main()












