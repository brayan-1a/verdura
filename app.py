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

df = load_and_select_data(supabase, "verduras", selected_columns)
st.write("Datos seleccionados:", df)
