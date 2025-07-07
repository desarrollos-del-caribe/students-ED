# Predicciones
import pandas as pd
from .excel_service import load_dataset
from .ml_service import (get_cached_sleep_model, get_cached_academic_model, train_academic_performance_risk_model, 
train_social_media_addiction_model, train_decision_tree_model, train_kmeans_model)

from utils.helpers import (save_plot_image_with_timestamp, clean_old_graphs, calculate_addicted_score, 
calculate_affects_academic, calculate_mental_health_score, classify_mental_health,
classify_addiction_score, classify_academic_impact, classify_platform_risk, 
classify_social_media_usage, classify_conflicts, get_personal_recommendations, classify_sleep_quality
)

from .graph_service import plot_mental_health_comparison, plot_addiction_risk_bar, plot_student_performance_comparison, plot_academic_risk_pie, plot_addiction_by_country, plot_correlation_heatmap
import seaborn as sns
import logging
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os
from typing import Dict, Any

GRAPH_DIR = os.path.join(os.path.dirname(__file__), '../static/graphs')

logger = logging.getLogger(__name__)

def analyze_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recibe los datos del usuario y genera un análisis completo:
    - Puntajes (adicción, salud mental, afectación académica)
    - Clasificaciones descriptivas
    - Recomendaciones personalizadas
    """

    try:
        # Extracción segura
        usage = float(user_data.get("social_media_usage", 0))
        sleep = float(user_data.get("sleep_hours_per_night", 0))
        conflicts = int(user_data.get("conflicts_over_social_media", 0))
        platform = user_data.get("main_platform", "")

        # Cálculos
        addicted_score = calculate_addicted_score(usage, conflicts)
        affects_academic = calculate_affects_academic(addicted_score, sleep)
        mental_score = calculate_mental_health_score(usage, sleep, conflicts, addicted_score, affects_academic)

        # Clasificaciones
        addiction_level = classify_addiction_score(addicted_score)
        sleep_quality = classify_sleep_quality(sleep)
        academic_impact = classify_academic_impact(affects_academic)
        mental_health_desc = classify_mental_health(mental_score)
        platform_risk = classify_platform_risk(platform)
        usage_risk = classify_social_media_usage(usage)
        conflict_level = classify_conflicts(conflicts)

        # Recomendaciones personalizadas
        recommendations_data = get_personal_recommendations({
            "addicted_score": addicted_score,
            "sleep_hours": sleep,
            "affects_academic": affects_academic,
            "usage_hours": usage,
            "conflicts": conflicts,
            "platform": platform
        })

        return {
            "addicted_score": addicted_score,
            "mental_health_score": mental_score,
            "affects_academic_performance": affects_academic,
            "classifications": {
                "mental_health": mental_health_desc,
                "addiction": addiction_level,
                "sleep": sleep_quality,
                "academic_impact": academic_impact,
                "platform": platform_risk,
                "usage": usage_risk,
                "conflicts": conflict_level
            },
            "recommendations": recommendations_data["recommendations"],
            "risk_factors": recommendations_data["risk_factors"]
        }

    except Exception as e:
        logger.error(f"Error en analyze_user_data: {str(e)}")
        return {"error": str(e)}

def predict_sleep_hours(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predice las horas de sueño basadas en el uso de redes sociales.
    """
    try:
        social_media_usage = float(user_data.get("social_media_usage", 0))
        model, scaler = get_cached_sleep_model()

        input_df = pd.DataFrame([[social_media_usage]], columns=["Avg_Daily_Usage_Hours"])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # Estadísticas del dataset
        df = load_dataset()
        df = df[
            (df["Avg_Daily_Usage_Hours"] >= 1) & (df["Avg_Daily_Usage_Hours"] <= 10) &
            (df["Sleep_Hours_Per_Night"] >= 3) & (df["Sleep_Hours_Per_Night"] <= 12)
        ]

        dataset_stats = {
            "avg_social_media_usage": round(float(df["Avg_Daily_Usage_Hours"].mean()), 2),
            "avg_sleep_hours_per_night": round(float(df["Sleep_Hours_Per_Night"].mean()), 2)
        }

        # Obtener puntos del dataset para el gráfico
        scatter_points = [
            {"x": float(row["Avg_Daily_Usage_Hours"]), "y": float(row["Sleep_Hours_Per_Night"])}
            for _, row in df.iterrows()
        ]

        # Obtener parámetros de la regresión lineal
        regression_line = {
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_)
        }

        return {
            "predicted_sleep_hours_per_night": round(float(prediction), 2),
            "dataset_stats": dataset_stats,
            "message": (
                f"Con un uso diario de {social_media_usage} horas, se estima que duermes aproximadamente "
                f"{round(float(prediction), 2)} horas por noche."
            ),
            "sleep_classification": classify_sleep_quality(prediction),
            "scatter_points": scatter_points[:100],  # Limitar a 100 puntos para rendimiento
            "regression_line": regression_line
        }

    except Exception as e:
        logger.error(f"Error en predict_sleep_hours: {str(e)}")
        return {
            "predicted_sleep_hours_per_night": 0,
            "dataset_stats": {},
            "error": str(e),
            "scatter_points": [],
            "regression_line": {"slope": 0, "intercept": 0}
        }

def predict_academic_impact(user_data):
    """
    Usa el modelo de regresión logística entrenado para predecir si el rendimiento académico
    del usuario se ve afectado por su estilo de vida digital.
    """
    try:
        usage = float(user_data.get("social_media_usage", 0))
        sleep = float(user_data.get("sleep_hours_per_night", 0))
        conflicts = int(user_data.get("conflicts_over_social_media", 0))

        model, scaler = get_cached_academic_model()
        input_df = pd.DataFrame([[usage, sleep, conflicts]], columns=[
            "Avg_Daily_Usage_Hours",
            "Sleep_Hours_Per_Night",
            "Conflicts_Over_Social_Media"
        ])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        return {
            "affects_academic_performance": int(prediction),
            "academic_impact_classification": classify_academic_impact(int(prediction))
        }

    except Exception as e:
        logger.error(f"Error en predict_academic_impact: {str(e)}")
        return {
            "affects_academic_performance": 0,
            "academic_impact_classification": "No se pudo predecir el impacto académico.",
            "error": str(e)
        }
        
#Predecir el riesgo de que el rendimiento academico se vea afectado pos: horas de uso, salud mental y horas de sueño
#Regresión logística	
def academic_performance_risk(user_data):
    try:
        # Extracción segura de datos
        usage = float(user_data.get("social_media_usage", 0))
        sleep = float(user_data.get("sleep_hours_per_night", 0))
        conflicts = int(user_data.get("conflicts_over_social_media", 0))

        # Cálculo del puntaje de adicción y salud mental
        addicted_score = calculate_addicted_score(usage, conflicts)
        affects_academic = calculate_affects_academic(addicted_score, sleep)
        mental_score = calculate_mental_health_score(usage, sleep, conflicts, addicted_score, affects_academic)
        
        df = load_dataset()
        model, scaler = train_academic_performance_risk_model(df)

        input_df = pd.DataFrame([{
            "Avg_Daily_Usage_Hours": usage,
            "Sleep_Hours_Per_Night": sleep,
            "Mental_Health_Score": mental_score
        }])
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        dataset_stats = {
            "avg_usage_hours": round(float(df["Avg_Daily_Usage_Hours"].mean()), 2),
            "avg_sleep_hours": round(float(df["Sleep_Hours_Per_Night"].mean()), 2),
            "avg_mental_health_score": round(float(df["Mental_Health_Score"].mean()), 2),
            "avg_academic_impact": round(float(df["Affects_Academic_Performance"].mean()), 2)
        }

        graph_data = plot_academic_risk_pie(probability)

        return {
            "risk": "Alto" if prediction == 1 else "Bajo",
            "probability": round(float(probability), 4),
            "dataset_stats": dataset_stats,
            "graph_data": graph_data
        }

    except Exception as e:
        logger.error(f"Error en academic_performance_risk: {str(e)}")
        return {
            "risk": "Bajo",
            "probability": 0,
            "dataset_stats": {},
            "error": str(e)
        }
 
#Usa los mdoelos de social_media_addiction_risk y academic_performance_risk para devolver predicciones  
#Se convirtieron valores del dataframe a tipos nativos python para evitar errores de serialización 
#se valido que sean compatibles con jsonify y se agregaron valores por defecto para evitar problemas de KeyError.
def student_performance_prediction(student_id):
    try:
        df = load_dataset()

        if "Student_ID" not in df.columns:
            return {"error": "Columna 'Student_ID' no encontrada en el dataset."}

        student_data = df[df["Student_ID"] == student_id]
        if student_data.empty:
            return {"error": f"Estudiante con ID {student_id} no encontrado"}

        usage = float(student_data["Avg_Daily_Usage_Hours"].iloc[0])
        addicted_score = float(student_data["Addicted_Score"].iloc[0])
        mental_health = float(student_data["Mental_Health_Score"].iloc[0])
        conflicts = float(student_data["Conflicts_Over_Social_Media"].iloc[0])
        sleep = float(student_data["Sleep_Hours_Per_Night"].iloc[0])
        academic_impact = float(student_data["Affects_Academic_Performance"].iloc[0])

        addiction_pred = social_media_addiction_risk(usage, addicted_score, mental_health, conflicts)
        academic_pred = academic_performance_risk(usage, sleep, mental_health)

        dataset_stats = {
            "avg_addicted_score": round(float(df["Addicted_Score"].mean()), 2),
            "avg_academic_impact": round(float(df["Affects_Academic_Performance"].mean()), 2),
            "student_addicted_score": round(addicted_score, 2),
            "student_academic_impact": round(academic_impact, 2)
        }

        graph_data = plot_student_performance_comparison(
            addicted_score, dataset_stats["avg_addicted_score"],
            academic_impact, dataset_stats["avg_academic_impact"]
        )

        return {
            "id": int(student_id),
            "addiction_risk": str(addiction_pred["risk"]),
            "addiction_probabilities": {
                "No adicción": float(addiction_pred["probabilities"].get("No adicción", 0)),
                "Adicción": float(addiction_pred["probabilities"].get("Adicción", 0))
            },
            "academic_risk": str(academic_pred["risk"]),
            "academic_risk_probability": float(academic_pred["probability"]),
            "dataset_stats": {
                "avg_addicted_score": float(dataset_stats["avg_addicted_score"]),
                "avg_academic_impact": float(dataset_stats["avg_academic_impact"]),
                "student_addicted_score": float(dataset_stats["student_addicted_score"]),
                "student_academic_impact": float(dataset_stats["student_academic_impact"])
            },
            "graph_data": graph_data
        }

    except Exception as e:
        logger.error(f"Error en student_performance_prediction: {str(e)}")
        return {"error": str(e)}

    
#Calcula y devuelve estadisticas de adicción por país        
def addiction_by_country(min_students=5):
    try:
        df = load_dataset()

        if "Country" not in df.columns or "Addicted_Score" not in df.columns:
            return {"error": "Columnas requeridas no encontradas"}

        country_stats = df.groupby("Country").agg({
            "Addicted_Score": ["mean", "count"],
            "Avg_Daily_Usage_Hours": "mean"
        }).reset_index()

        country_stats.columns = ["Country", "avg_addicted_score", "student_count", "avg_usage_hours"]
        country_stats = country_stats[country_stats["student_count"] >= min_students]

        result = {
            "countries": country_stats["Country"].tolist(),
            "avg_addicted_scores": [round(x, 2) for x in country_stats["avg_addicted_score"]],
            "student_counts": country_stats["student_count"].tolist(),
            "avg_usage_hours": [round(x, 2) for x in country_stats["avg_usage_hours"]]
        }

        dataset_stats = {
            "total_students": len(df),
            "avg_addicted_score": round(float(df["Addicted_Score"].mean()), 2),
            "avg_usage_hours": round(float(df["Avg_Daily_Usage_Hours"].mean()), 2)
        }

        graph_data = plot_addiction_by_country(result["countries"], result["avg_addicted_scores"])

        return {
            "country_data": result,
            "dataset_stats": dataset_stats,
            "graph_data": graph_data
        }

    except Exception as e:
        logger.error(f"Error en addiction_by_country: {str(e)}")
        return {"error": str(e)}

#Predice si un estudiante tiene riesgo alto o bajo de adicción según datos
#Random forest
def social_media_addiction_risk(user_data):
    try:
         # Extraer datos necesarios del usuario
        usage = float(user_data.get("social_media_usage", 0))
        sleep = float(user_data.get("sleep_hours_per_night", 0))
        conflicts = int(user_data.get("conflicts_over_social_media", 0))

        # Calcular puntajes requeridos
        addicted_score = calculate_addicted_score(usage, conflicts)
        affects_academic = calculate_affects_academic(addicted_score, sleep)
        mental_score = calculate_mental_health_score(usage, sleep, conflicts, addicted_score, affects_academic)

        df = load_dataset()
        model, scaler = train_social_media_addiction_model(df)

        input_df = pd.DataFrame([{
            "Avg_Daily_Usage_Hours": usage,
            "Addicted_Score": addicted_score,
            "Mental_Health_Score": mental_score,
            "Conflicts_Over_Social_Media": conflicts
        }])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        graph_data = plot_addiction_risk_bar(probabilities[0], probabilities[1])

        return {
            "risk": "Alto" if prediction == 1 else "Bajo",
            "probabilities": {
                "No adicción": round(probabilities[0], 3),
                "Adicción": round(probabilities[1], 3)
            },
            "graph_data": graph_data
        }

    except Exception as e:
        logger.error(f"Error en social_media_addiction_risk: {str(e)}")
        return {
            "risk": "Desconocido",
            "probabilities": {},
            "error": str(e)
        }

#Árbol de desición
def visualize_decision_tree(target_column):
    df = load_dataset()
    model, features = train_decision_tree_model(df, target_column)

    # Obtener texto del árbol
    from sklearn.tree import export_text
    tree_text = export_text(model, feature_names=features)

    return {
        "tree_text": tree_text,
        "target": target_column,
        "label": f"Árbol de decisión para {target_column}"
    }


#Clustering
def run_kmeans_clustering(n_clusters=3):
    df = load_dataset()
    model, columns, X_scaled = train_kmeans_model(df, n_clusters)

    df_plot = pd.DataFrame(X_scaled[:, :2], columns=["x", "y"])
    df_plot["cluster"] = model.labels_

    # Empaquetar los puntos como lista de objetos para el frontend
    points = df_plot.to_dict(orient="records")

    return {
        "clusters": n_clusters,
        "features_used": columns[:2],
        "points": points,
        "label": f"Clustering KMeans con {n_clusters} grupos"
    }
