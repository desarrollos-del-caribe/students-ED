# Entrenar los modelos
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

#Regresion lineal multiple
def train_mental_health_model(df): 
    """
    Entrena un modelo de regresión lineal múltiple para predecir la salud mental
    a partir del uso de redes sociales y estilo de vida.
    """

    # Variables predictoras (X) y variable objetivo (y)
    features = [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Addicted_Score",
        "Conflicts_Over_Social_Media",
        "Affects_Academic_Performance"
    ]
    target = "Mental_Health_Score"

    # Validar que las columnas existan
    if not all(col in df.columns for col in features + [target]):
        raise ValueError("Faltan columnas requeridas para entrenar el modelo de salud mental.")

    X = df[features]
    y = df[target]

    # Escalado (opcional, pero mejora la convergencia)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Separar datos para entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Entrenamiento
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluación rápida
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Entrenamiento completado: MSE = {mse:.2f}, R2 = {r2:.2f}")

    # Retornar el modelo y el scaler (para usar en predicción)
    return model, scaler

#Regresión lineal simple
def train_sleep_prediction_model(df): 
    """
    Entrena un modelo de regresión lineal simple para predecir las horas de sueño
    en función del uso diario de redes sociales.
    """
    # Validación
    if not all(col in df.columns for col in ["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"]):
        raise ValueError("Faltan columnas necesarias para entrenar el modelo de sueño.")

    X = df[["Avg_Daily_Usage_Hours"]]
    y = df["Sleep_Hours_Per_Night"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, scaler

#Regresión logística
def train_academic_impact_model(df): 
    """
    Entrena un modelo de regresión logística para predecir si el uso de redes afecta el rendimiento académico.
    """
    required_cols = [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Affects_Academic_Performance"
    ]
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Faltan columnas necesarias para entrenar el modelo académico.")

    X = df[["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Mental_Health_Score"]]
    y = df["Affects_Academic_Performance"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"📊 Modelo académico entrenado: Accuracy = {acc:.2f}")
    return model, scaler

#Modelo de regresión logistica
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

#Random Forest (clasificación)
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
    y = (df["Addicted_Score"] > 6).astype(int)  # Etiqueta binaria: riesgo si el score es alto (se puede ajustar el umbral)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Modelo de adicción entrenado. Accuracy: {acc:.2f}")

    return model, scaler

#Árbol de decisión
def train_decision_tree_model(df, target_column):
    """
    Entrena un árbol de decisión para clasificación.
    """
    features = [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Addicted_Score",
        "Conflicts_Over_Social_Media"
    ]

    if not all(col in df.columns for col in features + [target_column]):
        raise ValueError("Faltan columnas para entrenar árbol de decisión.")

    X = df[features]
    y = df[target_column]

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X, y)

    return model, features

#Clustering
def train_kmeans_model(df, n_clusters=3):
    """
    Aplica KMeans clustering sobre variables numéricas del dataset.
    """
    numeric_df = df.select_dtypes(include=['number']).drop(columns=["Student_ID"], errors="ignore")
    if numeric_df.empty:
        raise ValueError("No hay columnas numéricas para clustering.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    return kmeans, numeric_df.columns.tolist(), X_scaled