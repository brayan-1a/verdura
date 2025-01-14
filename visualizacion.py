import matplotlib.pyplot as plt

# Graficar ventas por fecha
def graficar_ventas(fechas, ventas):
    plt.figure(figsize=(10, 6))
    plt.plot(fechas, ventas, marker='o')
    plt.title('Ventas a lo largo del tiempo')
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad Vendida')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Graficar predicciones
def graficar_predicciones(predicciones):
    plt.figure(figsize=(10, 6))
    plt.plot(predicciones, marker='x', color='r')
    plt.title('Predicciones de Stock')
    plt.xlabel('Producto')
    plt.ylabel('Cantidad Predicha')
    plt.tight_layout()
    plt.show()
