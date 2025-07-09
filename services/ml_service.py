import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .excel_service import load_dataset

_cached_sleep_model = None
_cached_sleep_scaler = None
_cached_academic_model = None
_cached_academic_scaler = None

def get_cached_sleep_model():
    """
    Entrena un modelo de regresión lineal simple para predecir las horas de sueño
    en función del uso diario de redes sociales.
    """
    global _cached_sleep_model, _cached_sleep_scaler

    if _cached_sleep_model is not None and _cached_sleep_scaler is not None:
        return _cached_sleep_model, _cached_sleep_scaler

    df = load_dataset()
    
    if not all(col in df.columns for col in ["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"]):
        raise ValueError("Faltan columnas necesarias para entrenar el modelo de sueño.")

    df = df[
        (df["Avg_Daily_Usage_Hours"] >= 1) & (df["Avg_Daily_Usage_Hours"] <= 10) &
        (df["Sleep_Hours_Per_Night"] >= 3) & (df["Sleep_Hours_Per_Night"] <= 12)
    ]

    if df.empty:
        raise ValueError("No hay suficientes datos válidos para entrenar el modelo de sueño.")

    X = df[["Avg_Daily_Usage_Hours"]]
    y = df["Sleep_Hours_Per_Night"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.8, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    _cached_sleep_model = model
    _cached_sleep_scaler = scaler

    return model, scaler

def get_cached_academic_model():
    """
    Entrena y cachea un modelo de regresión logística para predecir si el uso de redes afecta el rendimiento académico.
    """
    global _cached_academic_model, _cached_academic_scaler

    if _cached_academic_model is not None and _cached_academic_scaler is not None:
        return _cached_academic_model, _cached_academic_scaler

    df = load_dataset()

    required_cols = [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Conflicts_Over_Social_Media",
        "Affects_Academic_Performance"
    ]

    if not all(col in df.columns for col in required_cols):
        raise ValueError("Faltan columnas necesarias para entrenar el modelo académico.")

    # Limpieza y filtrado simple (opcional)
    df = df[
        (df["Avg_Daily_Usage_Hours"] >= 1) &
        (df["Avg_Daily_Usage_Hours"] <= 10) &
        (df["Sleep_Hours_Per_Night"] >= 3) &
        (df["Sleep_Hours_Per_Night"] <= 12) &
        (df["Conflicts_Over_Social_Media"] >= 0)
    ]

    if df.empty:
        raise ValueError("No hay suficientes datos válidos para entrenar el modelo académico.")

    X = df[["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Conflicts_Over_Social_Media"]]
    y = df["Affects_Academic_Performance"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.8, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    _cached_academic_model = model
    _cached_academic_scaler = scaler

    return model, scaler

def train_academic_performance_risk_model(df): 
    """
    Entrena un modelo para predecir el riesgo académico (afectación del rendimiento).
    """
    required_cols = [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Affects_Academic_Performance"
    ]

    if not all(col in df.columns for col in required_cols):
        raise ValueError("Faltan columnas necesarias para entrenar el modelo de riesgo académico.")

    X = df[["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Mental_Health_Score"]]
    y = df["Affects_Academic_Performance"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Modelo de riesgo académico entrenado: Accuracy = {acc:.2f}")

    return model, scaler

def train_social_media_addiction_model(df): 	
    """
    Entrena un modelo para predecir riesgo de adicción a redes sociales.
    """
    required_cols = [
        "Avg_Daily_Usage_Hours",
        "Addicted_Score",
        "Mental_Health_Score",
        "Conflicts_Over_Social_Media"
    ]

    if not all(col in df.columns for col in required_cols):
        raise ValueError("Faltan columnas necesarias para entrenar el modelo de adicción.")

    X = df[required_cols]
    y = (df["Addicted_Score"] > 6).astype(int) 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Modelo de adicción entrenado. Accuracy: {acc:.2f}")

    return model, scaler

def train_decision_tree_model(df: pd.DataFrame, target_column: str) -> tuple[DecisionTreeClassifier, list[str]]:
    """
    Entrena un árbol de decisión para clasificación.

    Args:
        df: DataFrame con los datos de entrenamiento.
        target_column: Columna objetivo para la clasificación (por ejemplo, 'Addicted_Score').

    Returns:
        Tuple con el modelo entrenado y la lista de características usadas.

    Raises:
        ValueError: Si faltan columnas requeridas o si target_column no es válido.
    """
    features = [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Addicted_Score",
        "Conflicts_Over_Social_Media"
    ]

    if not all(col in df.columns for col in features + [target_column]):
        raise ValueError("Faltan columnas requeridas para entrenar el árbol de decisión.")
    if df[target_column].nunique() < 2:
        raise ValueError("La columna objetivo debe tener al menos dos clases para clasificación.")

    X = df[features]
    y = df[target_column]

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)

    return model, features

def train_kmeans_model(df, n_clusters=3):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    features = [
        "Age",
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Conflicts_Over_Social_Media"
    ]

    df = df.dropna(subset=features)
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X_scaled)

    return model, features, X_scaled, scaler
