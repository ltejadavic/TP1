from keras.models import load_model
import pickle
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Definición de la clase HybridModel que integra dos modelos: XGBoost y RNN.
class HybridModel(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_model, rnn_model):
        # Inicialización con ambos modelos entrenados.
        self.xgb_model = xgb_model
        self.rnn_model = rnn_model

    def predict_proba(self, X):
        # Obtener probabilidades de XGBoost.
        proba_xgb = self.xgb_model.predict_proba(X)[:, 1]
        # Preparar los datos para RNN y obtener sus probabilidades.
        X_rnn = np.reshape(X.values, (X.shape[0], 1, X.shape[1]))
        proba_rnn = self.rnn_model.predict(X_rnn).flatten()
        # Calcular el promedio de las probabilidades de ambos modelos.
        proba_avg = (proba_xgb + proba_rnn) / 2
        # Devolver un array con las probabilidades ajustadas.
        return np.vstack((1-proba_avg, proba_avg)).T

    def predict(self, X):
        # Calcular probabilidades y determinar la clase más probable.
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Función para cargar un modelo XGBoost desde un archivo.
def cargar_modelo_hxgboost(modelo_path):
    try:
        # Cargar el modelo XGBoost usando pickle.
        with open(modelo_path, 'rb') as archivo:
            modelo = pickle.load(archivo)
        return modelo
    except Exception as e:
        # Gestionar errores de carga y notificar.
        print(f"Error al cargar el modelo XGBoost: {e}")
        return None

# Función para cargar un modelo RNN desde un archivo.
def cargar_modelo_hrnn(modelo_path):
    try:
        # Cargar el modelo RNN usando Keras.
        modelo = load_model(modelo_path)
        return modelo
    except Exception as e:
        # Gestionar errores de carga y notificar.
        print(f"Error al cargar el modelo RNN: {e}")
        return None

# Función para realizar predicciones usando el modelo híbrido.
def predecir_con_modelo_hibrido(xgb_model_path, rnn_model_path, X):
    # Cargar modelos previamente entrenados.
    xgb_model = cargar_modelo_hxgboost(xgb_model_path)
    rnn_model = cargar_modelo_hrnn(rnn_model_path)

    # Verificar que ambos modelos se han cargado correctamente.
    if xgb_model is None or rnn_model is None:
        print("Error al cargar modelos")
        return None

    # Crear instancia del modelo híbrido y realizar predicciones.
    modelo_hibrido = HybridModel(xgb_model, rnn_model)
    predicciones = modelo_hibrido.predict(X)
    return predicciones