import streamlit as st
from config import get_supabase_client
from preprocess import load_and_select_data, clean_data, normalize_data, add_features
from model_train import train_decision_tree, train_random_forest, cross_validate_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

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

# Entrenar Árbol de Decisión
if st.button("Entrenar Árbol de Decisión"):
    try:
        model_tree, metrics_tree = train_decision_tree(df_features, target_col, feature_cols)
        st.write("Métricas del Árbol de Decisión:", metrics_tree)
    except Exception as e:
        st.error(f"Error al entrenar Árbol de Decisión: {e}")

# Entrenar Random Forest
if st.button("Entrenar Random Forest"):
    try:
        model_rf, metrics_rf = train_random_forest(df_features, target_col, feature_cols)
        st.write("Métricas del Random Forest:", metrics_rf)
    except Exception as e:
        st.error(f"Error al entrenar Random Forest: {e}")

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

# Visualización de predicciones para Árbol de Decisión
if st.button("Visualizar Predicciones - Árbol de Decisión"):
    try:
        model_tree, _ = train_decision_tree(df_features, target_col, feature_cols)
        X = df_features[feature_cols]
        y = df_features[target_col]
        y_pred = model_tree.predict(X)

        # Crear el gráfico
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', lw=2)
        ax.set_xlabel("Valores Reales")
        ax.set_ylabel("Predicciones")
        ax.set_title("Árbol de Decisión - Valores Reales vs Predicciones")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al visualizar las predicciones del Árbol de Decisión: {e}")

# Visualización de predicciones para Random Forest
if st.button("Visualizar Predicciones - Random Forest"):
    try:
        model_rf, _ = train_random_forest(df_features, target_col, feature_cols)
        X = df_features[feature_cols]
        y = df_features[target_col]
        y_pred = model_rf.predict(X)

        # Crear el gráfico
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', lw=2)
        ax.set_xlabel("Valores Reales")
        ax.set_ylabel("Predicciones")
        ax.set_title("Random Forest - Valores Reales vs Predicciones")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al visualizar las predicciones del Random Forest: {e}")






