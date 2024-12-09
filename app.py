import streamlit as st
import matplotlib.pyplot as plt
from config import get_supabase_client
from preprocess import load_and_select_data, clean_data, normalize_data, add_features
from model_train import train_decision_tree, train_random_forest, cross_validate_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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

# Agregar nuevas características
try:
    df_features = add_features(df_clean)
    st.write("Datos con nuevas características:", df_features.head())
except Exception as e:
    st.error(f"Error al agregar características: {e}")

# Definir columnas de entrada y objetivo
feature_cols = [
    "precio",
    "cantidad_vendida",
    "inventario_inicial",
    "desperdicio",
    "diferencia_inventario",
    "porcentaje_desperdicio",
    "es_fin_de_semana"
]
target_col = "inventario_final"

# Pestañas de navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Selecciona una pestaña", ["Predicción", "Gráficos de Stock", "Procedimiento"])

# Pestaña 1: Selección de datos para hacer la predicción
if page == "Predicción":
    st.header("Predicción de Stock de Verduras")
    st.sidebar.header("Selecciona los datos del producto")
    producto_seleccionado = st.sidebar.selectbox("Selecciona el producto", df["producto"].unique())

    # Filtrar datos para ese producto
    producto_data = df[df["producto"] == producto_seleccionado]

    # Mostrar los datos seleccionados
    st.write(f"Datos de {producto_seleccionado}:")
    st.write(producto_data)

    # Variables para hacer la predicción
    cantidad_actual = st.number_input("Cantidad actual en stock", min_value=0, value=int(producto_data["inventario_final"].iloc[0]))
    promocion = st.checkbox("Promoción activa")

    # Crear el vector de características para la predicción
    # Asegúrate de que estas son las mismas columnas que usaste en `feature_cols` para entrenar el modelo
    caracteristicas = [
        producto_data["precio"].iloc[0],  # Tomamos el precio del producto
        producto_data["cantidad_vendida"].iloc[0],  # Cantidad vendida
        producto_data["inventario_inicial"].iloc[0],  # Inventario inicial
        producto_data["desperdicio"].iloc[0],  # Desperdicio
        cantidad_actual,  # La cantidad actual en stock que ingresa el usuario
        promocion,  # Si hay promoción activa (True/False)
        producto_data["diferencia_inventario"].iloc[0],  # Diferencia de inventario calculada previamente
        producto_data["porcentaje_desperdicio"].iloc[0],  # Porcentaje de desperdicio calculado previamente
        producto_data["es_fin_de_semana"].iloc[0]  # Si es fin de semana o no
    ]

    # Al presionar el botón, hacer la predicción
    if st.button("Predecir cantidad recomendada para comprar"):
        try:
            # Asegúrate de que el modelo Random Forest esté entrenado
            model_rf, _ = train_random_forest(df_features, target_col, feature_cols)

            # Realizar la predicción usando el vector de características
            prediccion = model_rf.predict([caracteristicas])

            # Mostrar la recomendación
            st.write(f"Recomendación: Comprar {round(prediccion[0], 2)} unidades de {producto_seleccionado}")
        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")


# Pestaña 2: Gráficos de stock
elif page == "Gráficos de Stock":
    st.header("Gráficos de Inventario")
    fig, ax = plt.subplots()
    ax.bar(df["producto"], df["inventario_inicial"], label="Inventario Inicial")
    ax.bar(df["producto"], df["inventario_final"], label="Inventario Final", alpha=0.5)

    ax.set_xlabel("Producto")
    ax.set_ylabel("Cantidad")
    ax.set_title("Comparación de Inventario Inicial vs Final")
    ax.legend()

    st.pyplot(fig)

# Pestaña 3: Procedimiento y Resultados
elif page == "Procedimiento":
    st.header("Procedimiento y Resultados")
    st.write("Datos de Entrenamiento:")
    st.write(df_features.head())  # Muestra las primeras filas de los datos preprocesados

    st.write("Procedimiento de Predicción:")
    st.write("1. Se cargan los datos desde Supabase.")
    st.write("2. Se realizan transformaciones y limpieza de datos.")
    st.write("3. Se entrenan modelos de Árbol de Decisión y Random Forest.")
    st.write("4. Se predicen las cantidades recomendadas a comprar.")
    
    # Aquí puedes agregar más detalles de la metodología si lo deseas.

# Validación cruzada para Árbol de Decisión
if st.button("Validar Árbol de Decisión"):
    try:
        model_tree = DecisionTreeRegressor(random_state=42)
        mean_mse, std_mse = cross_validate_model(model_tree, df_features, target_col, feature_cols)
        st.write(f"Árbol de Decisión - MSE Promedio: {mean_mse:.4f}, Desviación Estándar: {std_mse:.4f}")
    except Exception as e:
        st.error(f"Error durante la validación cruzada del Árbol de Decisión: {e}")

# Validación cruzada para Random Forest
if st.button("Validar Random Forest"):
    try:
        model_rf = RandomForestRegressor(random_state=42, n_estimators=100)
        mean_mse, std_mse = cross_validate_model(model_rf, df_features, target_col, feature_cols)
        st.write(f"Random Forest - MSE Promedio: {mean_mse:.4f}, Desviación Estándar: {std_mse:.4f}")
    except Exception as e:
        st.error(f"Error durante la validación cruzada del Random Forest: {e}")







