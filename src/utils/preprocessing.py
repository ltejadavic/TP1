from flask import flash
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

def realizar_preprocesamiento(file_path, base_folder):
    # Cargar el dataset
    df = pd.read_csv(file_path)

    # Descartar la primera columna (código de cliente)
    df = df.iloc[:, 1:]

    # Identificar la columna objetivo
    target_col = df.columns[-1]

    # Estados válidos para análisis de Churn
    valid_states = ['Churned', 'Stayed', 'Yes', 'No', '1', '0']

    # Filtrar filas con estados válidos
    df = df[df[target_col].isin(valid_states)]

    # Verificar si quedan al menos dos estados únicos
    unique_states = df[target_col].unique()
    if len(unique_states) < 2:
        flash("No se encontraron estados válidos para el análisis de churn")
        return None

    # Transformación a 'Churn' binaria
    df['Churn'] = df[target_col].apply(lambda x: 1 if str(x).lower() in ['churned', 'yes', '1'] else 0)
    churn_data = df['Churn']
    df.drop(columns=[target_col], inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Reidentificar columnas numéricas y categóricas después de eliminar la columna 'Churn'
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Umbral para el número máximo de valores únicos en una columna categórica
    unique_values_threshold = 10  # Ajusta este valor según sea necesario

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

    # Resetear los índices de 'df_preprocessed' antes de añadir la columna 'Churn'
    df_preprocessed.reset_index(drop=True, inplace=True)

    # Añadir la columna 'Churn' de nuevo
    df_preprocessed['Churn'] = churn_data.values

    # g. Normalización
    scaler = StandardScaler()
    cols_to_scale = df_preprocessed.columns.difference(['Churn'])
    df_preprocessed[cols_to_scale] = scaler.fit_transform(df_preprocessed[cols_to_scale])

    # 2. Selección de características
    xgb_for_feature_selection = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_for_feature_selection.fit(df_preprocessed.drop('Churn', axis=1), df_preprocessed['Churn'])
    threshold = 0.01
    selected_features = df_preprocessed.drop('Churn', axis=1).columns[xgb_for_feature_selection.feature_importances_ > threshold].tolist()

    # 3. Balanceo con SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(df_preprocessed[selected_features], df_preprocessed['Churn'])

    # 4. División de Conjunto
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

    # Guardar los conjuntos de datos en archivos separados
    x_train_file_path = os.path.join(base_folder, 'X_train_data.csv')
    x_test_file_path = os.path.join(base_folder, 'X_test_data.csv')
    y_train_file_path = os.path.join(base_folder, 'y_train_data.csv')
    y_test_file_path = os.path.join(base_folder, 'y_test_data.csv')

    X_train.to_csv(x_train_file_path, index=False)
    X_test.to_csv(x_test_file_path, index=False)
    y_train.to_csv(y_train_file_path, index=False)
    y_test.to_csv(y_test_file_path, index=False)

    return df_preprocessed, selected_features, df_preprocessed.shape[0], len(selected_features), x_train_file_path, x_test_file_path, y_train_file_path, y_test_file_path
