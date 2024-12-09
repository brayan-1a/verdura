import pandas as pd

def load_and_select_data(supabase_client, table_name, selected_columns):
    """Carga datos desde Supabase y selecciona solo las columnas relevantes."""
    response = supabase_client.table(table_name).select(",".join(selected_columns)).execute()
    return pd.DataFrame(response.data)

def clean_data(df):
    """Realiza limpieza básica de los datos."""
    # Manejar valores nulos
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Reemplazar nulos en columnas numéricas con la media
    df.fillna("Sin datos", inplace=True)  # Reemplazar nulos en texto con un valor predeterminado

    # Eliminar columnas innecesarias (si las hay)
    if "nombre_cliente" in df.columns:
        df.drop(columns=["nombre_cliente"], inplace=True)

    return df

def normalize_data(df, columns):
    """Normaliza columnas específicas."""
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

