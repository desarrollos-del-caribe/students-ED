# Entrenar los modelos

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def train_social_media_addiction_model(df):
    required_cols = ['avg_daily_used_hours', 'addicted_score', 'mental_health_score', 'conflicts_over_social_media']

    if not all(col in df.columns for col in required_cols):
        logger.error("Columnas requeridas no encontradas para entrenamiento")
        raise ValueError("Columnas requeridas no encontradas para entrenamiento")

    def clasificar_riesgo(row):
        if row['avg_daily_used_hours'] > 5 and row['addicted_score'] >= 7 and row['conflicts_over_social_media'] >= 2:
            return "Alto"
        elif row['addicted_score'] >= 5 or row['mental_health_score'] <= 5:
            return "Medio"
        else:
            return "Bajo"

    df['Riesgo_Adiccion'] = df.apply(clasificar_riesgo, axis=1)

    X = df[required_cols]
    y = df['Riesgo_Adiccion']

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    return model