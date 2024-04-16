from flask import flash
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

def realizar_preprocesamiento_pred(file_path, x_train_path, output_directory):
    # Cargar el dataset
    df = pd.read_csv(file_path)
    # Descartar la primera columna si es necesario (dependiendo de la estructura de tu dataset)
    df = df.iloc[:, 1:]
     # Cargar el X_train para comparar las columnas después del preprocesamiento
    X_train = pd.read_csv(x_train_path)

    # Identificar columnas numéricas y categóricas
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Umbral para el número máximo de valores únicos en una columna categórica
    unique_values_threshold = 10

    def group_categories(series, threshold=0.05):
        """
        Agrupa las categorías menos frecuentes de una serie en una categoría 'Other'.
        
        Args:
        series (pd.Series): La serie cuyas categorías se van a agrupar.
        threshold (float): El umbral de frecuencia para agrupar categorías. Por defecto es 0.05.
        
        Returns:
        pd.Series: Una serie con las categorías agrupadas.
        """
        # Calcular las frecuencias de las categorías
        freq = series.value_counts(normalize=True)
        
        # Identificar categorías que son menos frecuentes que el umbral
        small_categories = freq[freq < threshold].index
        
        # Reemplazar categorías menos frecuentes con 'Other'
        return series.apply(lambda x: 'Other' if x in small_categories else x)

    # Identificar columnas categóricas que requieren agrupación y aplicarla
    for col in cat_cols:
        if df[col].nunique() > unique_values_threshold:
            df[col + '_Grouped'] = group_categories(df[col])
            df.drop(col, axis=1, inplace=True)

    # Actualizar la lista de columnas categóricas
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']

    # c. Eliminar columnas con más del 50% de datos faltantes
    threshold = len(df) // 2
    df = df.dropna(thresh=threshold, axis=1)

    # Actualizar cat_cols después de la eliminación
    cat_cols = [col for col in cat_cols if col in df.columns]

    # d. Imputar valores atípicos y valores faltantes
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].apply(lambda x: df[col].median() if (x < Q1 - 1.5 * IQR) or (x > Q3 + 1.5 * IQR) else x)

    # e. Preprocesamiento de variables categóricas y numéricas
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # f. Aplicar el preprocesamiento
    df_transformed = preprocessor.fit_transform(df)

    if hasattr(df_transformed, "toarray"):
        df_transformed = df_transformed.toarray()

    # Obtener nombres de las nuevas columnas de características categóricas
    new_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)

    # Combinar nombres de columnas numéricas y categóricas
    new_cols = num_cols.tolist() + new_cat_cols.tolist()

    # Crear un nuevo DataFrame con las columnas transformadas
    df_preprocessed = pd.DataFrame(df_transformed, columns=new_cols)

    # Normalización
    scaler = StandardScaler()
    df_preprocessed = pd.DataFrame(scaler.fit_transform(df_preprocessed), columns=df_preprocessed.columns)

    # Asegurar compatibilidad de columnas con X_train
    common_cols = [col for col in df_preprocessed.columns if col in X_train.columns]
    df_preprocessed = df_preprocessed[common_cols]

    # Agregar columnas faltantes de X_train si es necesario
    missing_cols = [col for col in X_train.columns if col not in df_preprocessed.columns]
    for col in missing_cols:
        df_preprocessed[col] = 0

    # Asegurarse de que el orden de las columnas sea el mismo que en X_train
    df_preprocessed = df_preprocessed[X_train.columns]

    # Datos adicionales para retornar
    num_rows = df_preprocessed.shape[0]
    num_columns = df_preprocessed.shape[1]

    return df_preprocessed, num_rows, num_columns