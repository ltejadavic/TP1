import os
from keras.models import load_model
from datetime import datetime

def guardar_modelo_rnn(modelo, usuario, nombre_dataset, accuracy, base_path="src/trained_rnn"):
    # Crear la carpeta base si no existe
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Crear la carpeta del usuario
    user_path = os.path.join(base_path, usuario)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    # Crear la carpeta de la fecha
    date_folder = datetime.now().strftime('%Y-%m-%d')
    date_path = os.path.join(user_path, date_folder)
    if not os.path.exists(date_path):
        os.makedirs(date_path)

    # Definir el nombre del archivo
    file_name = f"RNN_{nombre_dataset}_{accuracy:.2f}.h5"
    file_path = os.path.join(date_path, file_name)

    # Guardar el modelo usando el método save de Keras
    modelo.save(file_path)

    return file_path