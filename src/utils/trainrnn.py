import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

def realizar_rnn(x_train_path, x_test_path, y_train_path, y_test_path):
    # Cargar los datasets
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Preparar los datos para RNN
    X_train_rnn = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_rnn = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

    # Construir el modelo RNN con los hiperparámetros óptimos
    model = Sequential()
    model.add(LSTM(units=24, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]), return_sequences=True))
    model.add(LSTM(units=24, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Entrenamiento del modelo
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint('best_rnn_model.h5', save_best_only=True)
    ]

    model.fit(X_train_rnn, y_train, epochs=100, validation_split=0.2, callbacks=callbacks)

    # Evaluación del modelo
    loss, accuracy = model.evaluate(X_test_rnn, y_test)
    y_pred = (model.predict(X_test_rnn) > 0.5).astype("int32")

    # Calcular la matriz de confusión y el reporte de clasificación
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Extraer recall y f1-score del reporte
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    # Retornar el modelo y las métricas
    return model, accuracy, recall, f1, conf_matrix, report
