import streamlit as st
import pandas as pd
from supabase import create_client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import StandardScaler
from config import SUPABASE_URL, SUPABASE_KEY, MODEL_FEATURES, TARGET_VARIABLE, RF_PARAMS

# Función para cargar datos desde Supabase
def load_data_from_supabase():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table('verduras').select('*').execute()
    df = pd.DataFrame(response.data)
    return df

# Preprocesamiento de los datos
def preprocess_data(df):
    df_selected = df[MODEL_FEATURES + [TARGET_VARIABLE]].copy()
    df_selected['promocion'] = df_selected['promocion'].astype(int)
    df_selected['dia_semana'] = pd.Categorical(df_selected['dia_semana']).codes
    df_selected['mes'] = pd.Categorical(df_selected['mes']).codes
    
    # Imputar valores nulos y escalar los datos
    imputer = SimpleImputer(strategy='mean')
    df_selected[MODEL_FEATURES] = imputer.fit_transform(df_selected[MODEL_FEATURES])
    scaler = StandardScaler()
    df_selected[MODEL_FEATURES] = scaler.fit_transform(df_selected[MODEL_FEATURES])
    
    return df_selected, scaler

# Entrenamiento del modelo Random Forest
def train_random_forest(df):
    df_processed, scaler = preprocess_data(df)
    X = df_processed[MODEL_FEATURES]
    y = df_processed[TARGET_VARIABLE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    joblib.dump(model, 'models/random_forest_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')

    return model, scaler, mse, r2

# Predicción con el modelo entrenado
def predict_inventory(new_data):
    model = joblib.load('models/random_forest_model.joblib')
    scaler = joblib.load('models/scaler.joblib')

    new_data['promocion'] = new_data['promocion'].astype(int)
    new_data['dia_semana'] = pd.Categorical(new_data['dia_semana']).codes
    new_data['mes'] = pd.Categorical(new_data['mes']).codes

    new_data[MODEL_FEATURES] = scaler.transform(new_data[MODEL_FEATURES])
    prediction = model.predict(new_data[MODEL_FEATURES])
    return prediction

# Streamlit
def main():
    st.title('Predicción de Inventario de Verduras')

    if st.button('Cargar Datos'):
        df = load_data_from_supabase()
        st.write(df.head())

    if st.button('Entrenar Modelo'):
        df = load_data_from_supabase()
        model, scaler, mse, r2 = train_random_forest(df)
        st.success(f'Modelo entrenado con éxito')
        st.write(f'Error Cuadrático Medio: {mse}')
        st.write(f'R²: {r2}')

    st.header('Predecir Inventario')
    new_data = {}
    for feature in MODEL_FEATURES:
        new_data[feature] = st.text_input(f'Ingresa {feature}')

    if st.button('Predecir'):
        new_df = pd.DataFrame([new_data])
        prediction = predict_inventory(new_df)
        st.write(f'Inventario Final Predicho: {prediction[0]}')

if __name__ == '__main__':
    main()




