import pandas as pd

def load_and_select_data(supabase_client, table_name, selected_columns):
    """Carga datos desde Supabase y selecciona solo las columnas relevantes."""
    response = supabase_client.table(table_name).select(",".join(selected_columns)).execute()
    return pd.DataFrame(response.data)

