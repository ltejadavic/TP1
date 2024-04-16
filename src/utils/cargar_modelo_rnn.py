from keras.models import load_model

def cargar_modelo_rnn(modelo_path):
    """
    Carga un modelo de Keras desde un archivo .h5.
    """
    try:
        modelo = load_model(modelo_path)
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None