# Predicciones
import pandas as pd
from .excel_service import load_dataset
from .ml_service import train_mental_health_model, train_sleep_prediction_model, train_academic_impact_model, train_academic_performance_risk_model, train_social_media_addiction_model, train_decision_tree_model, train_kmeans_model
from utils.helpers import save_plot_image_with_timestamp, clean_old_graphs
import seaborn as sns
import logging
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

GRAPH_DIR = os.path.join(os.path.dirname(__file__), '../static/graphs')

logger = logging.getLogger(__name__)

def predict_mental_health_score(usage_hours, sleep_hours, addicted_score, conflicts_score, academic_impact):
    """
    Predice el score de salud mental con base en hábitos del usuario.
    """
    try:
        df = load_dataset()

        # Entrenar modelo
        model, scaler = train_mental_health_model(df)

        # Datos de entrada en el mismo orden que el entrenamiento
        input_df = pd.DataFrame([{
            "Avg_Daily_Usage_Hours": usage_hours,
            "Sleep_Hours_Per_Night": sleep_hours,
            "Addicted_Score": addicted_score,
            "Conflicts_Over_Social_Media": conflicts_score,
            "Affects_Academic_Performance": academic_impact
        }])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # Estadísticas del dataset
        dataset_stats = {
            "avg_usage_hours": round(float(df["Avg_Daily_Usage_Hours"].mean()), 2),
            "avg_sleep_hours": round(float(df["Sleep_Hours_Per_Night"].mean()), 2),
            "avg_addicted_score": round(float(df["Addicted_Score"].mean()), 2),
            "avg_conflicts_score": round(float(df["Conflicts_Over_Social_Media"].mean()), 2),
            "avg_academic_impact": round(float(df["Affects_Academic_Performance"].mean()), 2),
            "avg_mental_health_score": round(float(df["Mental_Health_Score"].mean()), 2)
        }

        return {
            "predicted_score": round(float(prediction), 2),
            "dataset_stats": dataset_stats
        }

    except Exception as e:
        logger.error(f"Error en predict_mental_health_score: {str(e)}")
        return {
            "predicted_score": 0,
            "dataset_stats": {},
            "error": str(e)
        }

def predict_sleep_hours(usage_hours):
    try:
        df = load_dataset()
        model, scaler = train_sleep_prediction_model(df)

        input_df = pd.DataFrame([[usage_hours]], columns=["Avg_Daily_Usage_Hours"])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        dataset_stats = {
            "avg_usage_hours": round(float(df["Avg_Daily_Usage_Hours"].mean()), 2),
            "avg_sleep_hours": round(float(df["Sleep_Hours_Per_Night"].mean()), 2)
        }

        return {
            "predicted_sleep_hours": round(float(prediction), 2),
            "dataset_stats": dataset_stats,
            "message": f"Con un uso diario de {usage_hours} horas, se estima que duermes aproximadamente {round(float(prediction), 2)} horas por noche."
        }

    except Exception as e:
        logger.error(f"Error en predict_sleep_hours: {str(e)}")
        return {
            "predicted_sleep_hours": 0,
            "dataset_stats": {},
            "error": str(e)
        }

def predict_academic_impact(usage_hours, sleep_hours, mental_health_score):
    try:
        df = load_dataset()
        model, scaler = train_academic_impact_model(df)

        input_df = pd.DataFrame([{
            "Avg_Daily_Usage_Hours": usage_hours,
            "Sleep_Hours_Per_Night": sleep_hours,
            "Mental_Health_Score": mental_health_score
        }])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # Probabilidad de que sea 1

        dataset_stats = {
            "avg_usage_hours": round(float(df["Avg_Daily_Usage_Hours"].mean()), 2),
            "avg_sleep_hours": round(float(df["Sleep_Hours_Per_Night"].mean()), 2),
            "avg_mental_health_score": round(float(df["Mental_Health_Score"].mean()), 2)
        }

        return {
            "impact": "Sí" if prediction == 1 else "No",
            "probability": round(float(probability), 4),
            "dataset_stats": dataset_stats,
            "message": (
                "Según tus hábitos, es probable que tu rendimiento académico "
                + ("esté siendo afectado." if prediction == 1 else "no esté siendo afectado significativamente.")
            )
        }

    except Exception as e:
        logger.error(f"Error en predict_academic_impact: {str(e)}")
        return {
            "impact": "No",
            "probability": 0,
            "dataset_stats": {},
            "error": str(e)
        }
#Predecir el riesgo de que el rendimiento academico se vea afectado pos: horas de uso, salud mental y horas de sueño
#Regresión logística	
def academic_performance_risk(usage_hours, sleep_hours, mental_health_score):
    """
    Predice el riesgo académico usando un modelo de regresión logística.
    """
    try:
        df = load_dataset()
        model, scaler = train_academic_performance_risk_model(df)

        input_df = pd.DataFrame([{
            "Avg_Daily_Usage_Hours": usage_hours,
            "Sleep_Hours_Per_Night": sleep_hours,
            "Mental_Health_Score": mental_health_score
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

        return {
            "risk": "Alto" if prediction == 1 else "Bajo",
            "probability": round(float(probability), 4),
            "dataset_stats": dataset_stats
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
def student_performance_prediction(student_id):
    """Predice el rendimiento académico y riesgo de adicción para un estudiante específico usando el dataset Excel."""
    try:
        df = load_dataset()

        if "Student_ID" not in df.columns:
            return {"error": "Columna 'Student_ID' no encontrada en el dataset."}

        student_data = df[df["Student_ID"] == student_id]
        if student_data.empty:
            return {"error": f"Estudiante con ID {student_id} no encontrado"}

        # Extraer valores del estudiante
        usage = student_data["Avg_Daily_Usage_Hours"].iloc[0]
        addicted_score = student_data["Addicted_Score"].iloc[0]
        mental_health = student_data["Mental_Health_Score"].iloc[0]
        conflicts = student_data["Conflicts_Over_Social_Media"].iloc[0]
        sleep = student_data["Sleep_Hours_Per_Night"].iloc[0]

        # Predicciones
        addiction_pred = social_media_addiction_risk(usage, addicted_score, mental_health, conflicts)
        academic_pred = academic_performance_risk(usage, sleep, mental_health)

        dataset_stats = {
            "avg_addicted_score": round(float(df["Addicted_Score"].mean()), 2),
            "avg_academic_impact": round(float(df["Affects_Academic_Performance"].mean()), 2),
            "student_addicted_score": round(float(addicted_score), 2),
            "student_academic_impact": round(float(student_data["Affects_Academic_Performance"].iloc[0]), 2)
        }

        return {
            "id": student_id,
            "addiction_risk": addiction_pred["risk"],
            "addiction_probabilities": addiction_pred["probabilities"],
            "academic_risk": academic_pred["risk"],
            "academic_risk_probability": academic_pred["probability"],
            "dataset_stats": dataset_stats
        }

    except Exception as e:
        logger.error(f"Error en student_performance_prediction: {str(e)}")
        return {"error": str(e)}
    
#Calcula y devuelve estadisticas de adicción por país        
def addiction_by_country(min_students=5):
    """
    Calcula el riesgo promedio de adicción por país basado en el dataset Excel.
    """
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

        return {
            "country_data": result,
            "dataset_stats": dataset_stats
        }

    except Exception as e:
        logger.error(f"Error en addiction_by_country: {str(e)}")
        return {"error": str(e)}

#Predice si un estudiante tiene riesgo alto o bajo de adicción según datos
#Random forest
def social_media_addiction_risk(usage_hours, addicted_score, mental_health_score, conflicts_score):
    """
    Predice el riesgo de adicción a redes sociales de un estudiante.
    """
    try:
        df = load_dataset()
        model, scaler = train_social_media_addiction_model(df)

        input_df = pd.DataFrame([{
            "Avg_Daily_Usage_Hours": usage_hours,
            "Addicted_Score": addicted_score,
            "Mental_Health_Score": mental_health_score,
            "Conflicts_Over_Social_Media": conflicts_score
        }])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        return {
            "risk": "Alto" if prediction == 1 else "Bajo",
            "probabilities": {
                "No adicción": round(probabilities[0], 3),
                "Adicción": round(probabilities[1], 3)
            }
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

    # Graficar
    plt.figure(figsize=(12, 6))
    plot_tree(model, feature_names=features, filled=True, class_names=True)

    # Limpiar imágenes viejas antes de guardar
    clean_old_graphs(GRAPH_DIR)

    return save_plot_image_with_timestamp(f"tree_{target_column}")

#Clustering
def run_kmeans_clustering(n_clusters=3):
    df = load_dataset()
    model, columns, X_scaled = train_kmeans_model(df, n_clusters)

    df_plot = pd.DataFrame(X_scaled[:, :2], columns=["Feature1", "Feature2"])
    df_plot["Cluster"] = model.labels_

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x="Feature1", y="Feature2", hue="Cluster", palette="Set2")

    clean_old_graphs(GRAPH_DIR)

    return save_plot_image_with_timestamp("kmeans_clusters")