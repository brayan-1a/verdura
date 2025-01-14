import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preparar_datos_modelo(df_ventas):
    """Prepara los datos para el entrenamiento del modelo"""
    
    # Convertir fechas a datetime
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    
    # Crear características para el modelo
    df_final = df_ventas.copy()
    df_final['dia_semana'] = df_final['fecha_venta'].dt.dayofweek
    df_final['mes'] = df_final['fecha_venta'].dt.month
    df_final['año'] = df_final['fecha_venta'].dt.year
    df_final['dia_mes'] = df_final['fecha_venta'].dt.day
    df_final['es_fin_de_semana'] = df_final['dia_semana'].isin([5, 6])
    
    # Agrupar ventas por producto, día de la semana y mes
    df_agrupado = df_final.groupby(
        ['producto_id', 'dia_semana', 'mes', 'año', 'dia_mes', 'es_fin_de_semana']
    )['cantidad_vendida'].sum().reset_index()
    
    # Normalizar la cantidad vendida
    scaler = StandardScaler()
    df_agrupado['cantidad_vendida'] = scaler.fit_transform(df_agrupado[['cantidad_vendida']])

    # Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X = df_agrupado.drop(columns=['cantidad_vendida'])
    y = df_agrupado['cantidad_vendida']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test









