import pickle

def cargar_modelo(modelo_path):
    """
    Carga un modelo de XGBoost desde un archivo .model.
    """
    try:
        with open(modelo_path, 'rb') as archivo:
            modelo = pickle.load(archivo)
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None