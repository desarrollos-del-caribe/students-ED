# Cargar EXCEL

import pandas as pd
import os

# Ruta al archivo Excel
EXCEL_PATH = os.path.join(os.path.dirname(__file__), '../data/excel/setData.xlsx')

# Dataset en caché para no recargar desde disco cada vez
_cached_df = None

def load_dataset():
    """Carga el archivo Excel una vez y reutiliza el DataFrame."""
    global _cached_df
    if _cached_df is not None:
        return _cached_df

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"El archivo {EXCEL_PATH} no existe.")
    
    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        raise ValueError(f"Error al leer el archivo Excel: {str(e)}")
    
    _cached_df = df
    return df

def get_features_and_target(target_column: str):
    """
    Extrae X (características) e y (etiqueta) del dataset, excluyendo Student_ID.
    - target_column: la columna objetivo para entrenamiento.
    """
    df = load_dataset()
    if target_column not in df.columns:
        raise ValueError(f"La columna '{target_column}' no está en el dataset.")
    
    # Excluir Student_ID si existe
    X = df.drop(columns=["Student_ID", target_column], errors='ignore')
    y = df[target_column]
    return X, y

def get_all_columns():
    """Devuelve la lista de todas las columnas del dataset."""
    df = load_dataset()
    return df.columns.tolist()

def get_numerical_data():
    """Devuelve solo las columnas numéricas, útil para clustering como k-means."""
    df = load_dataset()
    return df.select_dtypes(include=['number'])