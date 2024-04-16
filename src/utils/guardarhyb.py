import os
from keras.models import load_model
from datetime import datetime
import pickle

def guardar_modelo_hibrido(xgb_model, rnn_model, usuario, nombre_dataset, accuracy, base_path="src/trained_hybrid"):
    # Crear la carpeta base si no existe
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Crear la carpeta del usuario
    user_path = os.path.join(base_path, usuario)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    # Crear la carpeta de la fecha
    date_folder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    date_path = os.path.join(user_path, date_folder)
    if not os.path.exists(date_path):
        os.makedirs(date_path)

    # Definir el nombre del archivo para cada modelo
    xgb_file_name = f"XGBoost_{nombre_dataset}_{accuracy:.2f}.model"
    rnn_file_name = f"RNN_{nombre_dataset}_{accuracy:.2f}.h5"

    # Rutas de los archivos
    xgb_file_path = os.path.join(date_path, xgb_file_name)
    rnn_file_path = os.path.join(date_path, rnn_file_name)

    # Guardar el modelo XGBoost
    with open(xgb_file_path, 'wb') as archivo:
        pickle.dump(xgb_model, archivo)

    # Guardar el modelo RNN
    rnn_model.save(rnn_file_path)


    return xgb_file_path, rnn_file_path