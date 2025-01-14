import streamlit as st
import matplotlib.pyplot as plt

# Mostrar gráfico de ventas
def mostrar_grafico_ventas(df):
    st.subheader("Gráfico de Ventas")
    ventas_totales = df.groupby('nombre_producto')['cantidad_vendida'].sum()
    ventas_totales.plot(kind='bar', figsize=(10, 6))
    plt.title('Ventas Totales por Producto')
    plt.ylabel('Cantidad Vendida')
    st.pyplot()

# Mostrar gráfico de predicciones
def mostrar_grafico_predicciones(df):
    st.subheader("Gráfico de Predicciones")
    # Aquí se podría hacer una predicción sobre los productos y mostrar un gráfico similar
    predicciones = df['cantidad_vendida'] * 0.9  # Ejemplo de predicción arbitraria
    df['predicciones'] = predicciones
    df.set_index('nombre_producto', inplace=True)
    df[['cantidad_vendida', 'predicciones']].plot(kind='bar', figsize=(10, 6))
    plt.title('Predicción vs Ventas Reales')
    plt.ylabel('Cantidad')
    st.pyplot()

