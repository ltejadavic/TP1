import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

# Función para entrenar el modelo XGBoost
def entrenar_xgboost(x_train_path, y_train_path):
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    xgb_model = XGBClassifier(
        subsample=1.0,
        scale_pos_weight=1,
        reg_lambda=1.0,
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        gamma=0,
        colsample_bytree=0.9,
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

# Función para entrenar el modelo RNN
def entrenar_rnn(x_train_path, y_train_path):
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    X_train_rnn = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))

    rnn_model = Sequential()
    rnn_model.add(LSTM(24, return_sequences=True, input_shape=(1, X_train.shape[1])))
    rnn_model.add(LSTM(24, return_sequences=False))
    rnn_model.add(Dense(1, activation='sigmoid'))

    rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    rnn_model.fit(X_train_rnn, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])
    
    return rnn_model

# Clase para el modelo híbrido utilizando soft voting
class HybridModel(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_model, rnn_model):
        self.xgb_model = xgb_model
        self.rnn_model = rnn_model

    def predict_proba(self, X):
        proba_xgb = self.xgb_model.predict_proba(X)[:, 1]
        X_rnn = np.reshape(X.values, (X.shape[0], 1, X.shape[1]))
        proba_rnn = self.rnn_model.predict(X_rnn).flatten()

        proba_avg = (proba_xgb + proba_rnn) / 2
        return np.vstack((1-proba_avg, proba_avg)).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    # Función principal para entrenar el modelo híbrido
def realizar_hyb(x_train_path, x_test_path, y_train_path, y_test_path):
    xgb_model = entrenar_xgboost(x_train_path, y_train_path)
    rnn_model = entrenar_rnn(x_train_path, y_train_path)

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    modelo_hibrido = HybridModel(xgb_model, rnn_model)
    predicciones = modelo_hibrido.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)
    recall = recall_score(y_test, predicciones)
    f1 = f1_score(y_test, predicciones)
    conf_matrix = confusion_matrix(y_test, predicciones)
    report = classification_report(y_test, predicciones)

    print('Rendimiento del modelo híbrido:')
    print(report)

    return xgb_model, rnn_model, accuracy, recall, f1, conf_matrix, report