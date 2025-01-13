import pickle

# Cargar el modelo entrenado
with open('modelo_entrenado.pkl', 'rb') as f:
    model = pickle.load(f)

def make_prediction(producto, descuento, temperatura, humedad):
    # Aquí deberías hacer el preprocesamiento necesario según lo que el usuario ingrese
    # Para este ejemplo, usamos los valores directamente
    X_new = [[descuento, temperatura, humedad]]  # Características del producto (según lo que elijas)
    
    # Realizar la predicción
    prediccion = model.predict(X_new)[0]
    
    return prediccion



