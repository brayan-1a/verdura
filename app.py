import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Tuple, Any
from datetime import datetime
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseConnection:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)
    
    def fetch_table_data(self, table_name: str) -> pd.DataFrame:
        try:
            query = self.client.table(table_name).select("*").execute()
            if 'data' not in query:
                raise ValueError(f"No se encontraron datos en la tabla {table_name}")
            return pd.DataFrame(query['data'])
        except Exception as e:
            logger.error(f"Error al cargar datos de {table_name}: {str(e)}")
            raise

class DataProcessor:
    @staticmethod
    def load_and_merge_data(supabase_conn: SupabaseConnection) -> pd.DataFrame:
        """Carga y combina datos de todas las tablas"""
        try:
            # Cargar datos de cada tabla
            tables = {
                'ventas': supabase_conn.fetch_table_data("ventas"),
                'productos': supabase_conn.fetch_table_data("productos"),
                'desperdicio': supabase_conn.fetch_table_data("desperdicio"),
                'promociones': supabase_conn.fetch_table_data("promociones"),
                'clima': supabase_conn.fetch_table_data("condiciones_climaticas")
            }
            
            # Combinar los datos
            df = tables['ventas']
            for table_name, table_df in tables.items():
                if table_name == 'ventas':
                    continue
                if table_name == 'clima':
                    df = pd.merge(df, table_df, left_on="fecha_venta", right_on="fecha", how="left")
                else:
                    df = pd.merge(df, table_df, on="producto_id", how="left")
            
            return DataProcessor.clean_data(df)
        except Exception as e:
            logger.error(f"Error en el procesamiento de datos: {str(e)}")
            raise

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y preprocesa los datos"""
        df = df.copy()
        
        # Convertir fechas
        df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
        
        # Rellenar valores faltantes
        numeric_columns = ['cantidad_perdida', 'descuento_aplicado', 'temperatura', 'humedad']
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].mean())
        
        # Agregar características temporales
        df['dia_semana'] = df['fecha_venta'].dt.dayofweek
        df['mes'] = df['fecha_venta'].dt.month
        df['temporada'] = pd.cut(df['mes'], 
                               bins=[0, 3, 6, 9, 12], 
                               labels=['invierno', 'primavera', 'verano', 'otoño'])
        
        return df

class ModelTrainer:
    def __init__(self):
        self.models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression()
        }
        self.scaler = StandardScaler()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara las características para el entrenamiento"""
        features = ['cantidad_vendida', 'descuento_aplicado', 'temperatura', 
                   'humedad', 'cantidad_perdida', 'dia_semana', 'mes']
        
        X = df[features]
        y = df['cantidad_vendida']
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Entrena y evalúa múltiples modelos"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results[name] = {
                    "Modelo": name,
                    "MSE": mean_squared_error(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "R2": r2_score(y_test, y_pred)
                }
            except Exception as e:
                logger.error(f"Error en el entrenamiento del modelo {name}: {str(e)}")
                
        return results

    def save_model(self, model_name: str, model_path: str = "models") -> str:
        """Guarda el modelo entrenado y el scaler"""
        os.makedirs(model_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_file = os.path.join(model_path, f"{model_name}_{timestamp}.pkl")
        scaler_file = os.path.join(model_path, f"scaler_{timestamp}.pkl")
        
        joblib.dump(self.models[model_name], model_file)
        joblib.dump(self.scaler, scaler_file)
        
        return model_file

def main():
    st.title("Sistema de Predicción de Stock de Verduras")
    
    # Configuración de Supabase
    supabase_conn = SupabaseConnection(
        url="https://odlosqyzqrggrhvkdovj.supabase.co",
        key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck"
    )
    
    if st.button("Entrenar Modelo"):
        try:
            with st.spinner("Cargando y procesando datos..."):
                # Cargar y procesar datos
                df = DataProcessor.load_and_merge_data(supabase_conn)
                st.success("Datos cargados y procesados correctamente")
                
                # Entrenar modelos
                trainer = ModelTrainer()
                X, y = trainer.prepare_features(df)
                results = trainer.train_and_evaluate(X, y)
                
                # Mostrar resultados
                st.subheader("Resultados de la Evaluación")
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df)
                
                # Identificar mejor modelo
                best_model = max(results.items(), key=lambda x: x[1]['R2'])[0]
                st.success(f"Mejor modelo: {best_model} (R² = {results[best_model]['R2']:.4f})")
                
                # Opción para guardar el modelo
                if st.button("Guardar Mejor Modelo"):
                    model_path = trainer.save_model(best_model)
                    st.download_button(
                        label="Descargar Modelo",
                        data=open(model_path, 'rb'),
                        file_name=os.path.basename(model_path),
                        mime="application/octet-stream"
                    )
                    
        except Exception as e:
            st.error(f"Error durante el proceso: {str(e)}")
            logger.error(f"Error en la aplicación: {str(e)}")

if __name__ == "__main__":
    main()













