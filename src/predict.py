from supabase import create_client, Client
import pandas as pd
from config import SUPABASE_URL, SUPABASE_KEY
from model_training import train_model
from data_loading import load_data
from data_preprocessing import preprocess_data

def save_predictions(predictions, dates, productos):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    data = {'fecha': dates, 'producto': productos, 'cantidad_vendida_predicha': predictions, 'modelo': 'RandomForest'}
    for i in range(len(dates)):
        row = {key: data[key][i] for key in data}
        supabase.table('predicciones').insert(row).execute()

def main():
    model = train_model()
    df = load_data()
    df = preprocess_data(df)
    X = df.drop(columns=['cantidad_vendida', 'fecha', 'nombre_cliente', 'dia_semana', 'notas_adicionales'])
    dates = df['fecha'].tolist()
    productos = df['producto'].tolist()
    predictions = model.predict(X).tolist()
    save_predictions(predictions, dates, productos)

if __name__ == "__main__":
    main()
