import pickle

def make_prediction(producto, descuento, temperatura, humedad):
    # Cargar el modelo entrenado
    with open('modelo_entrenado.pkl', 'rb') as f:
        model = pickle.load(f)

    # Crear las características para la predicción
    X_new = [[descuento, temperatura, humedad]]  # Características del producto (según lo que elijas)

    # Realizar la predicción
    prediccion = model.predict(X_new)[0]
    
    return prediccion




