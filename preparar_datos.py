import pandas as pd
import numpy as np

def preparar_datos_modelo(df_ventas):
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    
    # Crear características para el modelo
    df_ventas['dia_semana'] = df_ventas['fecha_venta'].dt.dayofweek
    df_ventas['mes'] = df_ventas['fecha_venta'].dt.month
    
    # Crear nuevas columnas necesarias para el modelo
    df_ventas['tendencia'] = np.arange(len(df_ventas))
    df_ventas['es_fin_semana'] = df_ventas['dia_semana'].isin([5, 6]).astype(int)
    df_ventas['temporada'] = pd.cut(df_ventas['mes'], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3])
    
    # Verificar si las columnas necesarias están presentes
    print("Columnas disponibles en df_ventas antes de retornar:")
    print(df_ventas.columns)

    # Calcular la diferencia de inventario
    df_ventas['diferencia_inventario'] = df_ventas['inventario_inicial'] - df_ventas['inventario_final']
    
    # Verificar si las columnas de inventarios y desperdicio no tienen valores nulos
    df_ventas['inventario_inicial'] = df_ventas['inventario_inicial'].fillna(0)
    df_ventas['inventario_final'] = df_ventas['inventario_final'].fillna(0)
    df_ventas['cantidad_perdida'] = df_ventas['cantidad_perdida'].fillna(0)

    # Asegúrate de que todas las columnas necesarias existan
    columnas_necesarias = ['dia_semana', 'mes', 'tendencia', 'es_fin_semana', 'temporada', 'diferencia_inventario', 'cantidad_perdida']
    columnas_faltantes = [col for col in columnas_necesarias if col not in df_ventas.columns]
    
    if columnas_faltantes:
        print(f"Faltan las siguientes columnas: {columnas_faltantes}")
    
    # Agrupar datos
    df_agrupado = df_ventas.groupby(
        ['producto_id', 'dia_semana', 'mes']
    )['cantidad_vendida'].mean().reset_index()

    return df_agrupado
















