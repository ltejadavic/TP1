import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import os
from datetime import datetime

def realizar_xgboost(x_train_path, x_test_path, y_train_path, y_test_path):
    # Cargar los datos
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    # Definir el modelo con los hiperparámetros optimizados
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

    # Entrenar el modelo
    xgb_model.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = xgb_model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Información adicional (opcional)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Devolver el modelo y las métricas
    return xgb_model, accuracy, recall, f1, conf_matrix, report
