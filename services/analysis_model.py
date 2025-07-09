# Predicciones
import numpy as np
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
from sklearn.tree import export_text, plot_tree
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
        usage = float(user_data.get("social_media_usage", 0))
        sleep = float(user_data.get("sleep_hours_per_night", 0))
        conflicts = int(user_data.get("conflicts_over_social_media", 0))
        platform = user_data.get("main_platform", "")

        addicted_score = calculate_addicted_score(usage, conflicts)
        affects_academic = calculate_affects_academic(addicted_score, sleep)
        mental_score = calculate_mental_health_score(usage, sleep, conflicts, addicted_score, affects_academic)

        addiction_level = classify_addiction_score(addicted_score)
        sleep_quality = classify_sleep_quality(sleep)
        academic_impact = classify_academic_impact(affects_academic)
        mental_health_desc = classify_mental_health(mental_score)
        platform_risk = classify_platform_risk(platform)
        usage_risk = classify_social_media_usage(usage)
        conflict_level = classify_conflicts(conflicts)

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
    Predice las horas de sueño basadas en el uso de redes sociales, con ajustes para uso bajo y extremo.
    """
    try:
        social_media_usage = float(user_data.get("social_media_usage", 0))
        
        if not 0 <= social_media_usage <= 24:
            raise ValueError("El uso de redes sociales debe estar entre 0 y 24 horas.")

        model, scaler = get_cached_sleep_model()

        input_df = pd.DataFrame([[social_media_usage]], columns=["Avg_Daily_Usage_Hours"])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        if social_media_usage <= 4:
            prediction = max(float(prediction), 8.0)
        elif social_media_usage > 10:
            penalty = (social_media_usage - 10) * 0.5  
            prediction = max(0, prediction - penalty)
        elif social_media_usage > 5:
            penalty = (social_media_usage - 5) * 0.2  
            prediction = max(0, prediction - penalty)

        prediction = min(max(float(prediction), 0), 12)

        df = load_dataset()
        df = df[
            (df["Avg_Daily_Usage_Hours"] >= 0) & (df["Avg_Daily_Usage_Hours"] <= 24) &
            (df["Sleep_Hours_Per_Night"] >= 0) & (df["Sleep_Hours_Per_Night"] <= 12)
        ]

        dataset_stats = {
            "avg_social_media_usage": round(float(df["Avg_Daily_Usage_Hours"].mean()), 2),
            "avg_sleep_hours_per_night": round(float(df["Sleep_Hours_Per_Night"].mean()), 2)
        }

        scatter_points = [
            {"x": float(row["Avg_Daily_Usage_Hours"]), "y": float(row["Sleep_Hours_Per_Night"])}
            for _, row in df.iterrows()
        ]

        regression_line = {
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_)
        }

        message = (
            f"Con un uso diario de {social_media_usage} horas de redes sociales, "
            f"se estima que duermes aproximadamente {round(prediction, 2)} horas por noche."
        )
        if social_media_usage <= 4:
            message += (
                " Tu bajo uso de redes sociales favorece un sueño adecuado."
            )
        elif social_media_usage > 10:
            message += (
                " Este nivel de uso es muy alto y probablemente reduce significativamente tus horas de sueño. "
                "Considera reducir el tiempo en redes sociales para mejorar tu descanso."
            )

        recommendations = []
        if prediction < 6:
            recommendations.append("Tu tiempo de sueño es bajo. Intenta reducir el uso de redes sociales antes de dormir.")
            recommendations.append("Establece una rutina de sueño consistente, evitando pantallas al menos 1 hora antes de acostarte.")
        elif prediction < 8:
            recommendations.append("Tu tiempo de sueño es aceptable, pero podrías beneficiarte de más descanso.")
            recommendations.append("Considera limitar el uso de redes sociales en la noche para mejorar la calidad del sueño.")
        if social_media_usage > 10:
            recommendations.append("El uso excesivo de redes sociales puede estar afectando tu salud. Usa aplicaciones para controlar tu tiempo en pantalla.")
        elif social_media_usage <= 4:
            recommendations.append("Tu bajo uso de redes sociales es positivo para tu descanso. Mantén una rutina de sueño saludable.")

        return {
            "predicted_sleep_hours_per_night": round(prediction, 2),
            "dataset_stats": dataset_stats,
            "message": message,
            "sleep_classification": classify_sleep_quality(prediction),
            "scatter_points": scatter_points[:100],  
            "regression_line": regression_line,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Error en predict_sleep_hours: {str(e)}")
        return {
            "predicted_sleep_hours_per_night": 0,
            "dataset_stats": {},
            "error": str(e),
            "scatter_points": [],
            "regression_line": {"slope": 0, "intercept": 0},
            "recommendations": []
        }

def predict_academic_impact(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Usa el modelo de regresión logística entrenado para predecir si el rendimiento académico
    del usuario se ve afectado por su estilo de vida digital.
    """
    try:
        usage = float(user_data.get("social_media_usage", 0))
        sleep = float(user_data.get("sleep_hours_per_night", 0))
        conflicts = int(user_data.get("conflicts_over_social_media", 0))

        if not 0 <= usage <= 24:
            raise ValueError("El uso de redes sociales debe estar entre 0 y 24 horas.")
        if not 0 <= sleep <= 12:
            raise ValueError("Las horas de sueño deben estar entre 0 y 12 horas.")
        if conflicts < 0:
            raise ValueError("Los conflictos deben ser un número entero no negativo.")

        model, scaler = get_cached_academic_model()
        input_df = pd.DataFrame([[usage, sleep, conflicts]], columns=[
            "Avg_Daily_Usage_Hours",
            "Sleep_Hours_Per_Night",
            "Conflicts_Over_Social_Media"
        ])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  

        df = load_dataset()
        df = df[
            (df["Avg_Daily_Usage_Hours"].between(0, 24)) &
            (df["Sleep_Hours_Per_Night"].between(0, 12)) &
            (df["Conflicts_Over_Social_Media"] >= 0) &
            (df["Affects_Academic_Performance"].isin([0, 1]))
        ]

        dataset_points = [
            {
                "x": float(row["Avg_Daily_Usage_Hours"]),
                "y": float(row["Sleep_Hours_Per_Night"]),
                "label": int(row["Affects_Academic_Performance"])
            }
            for _, row in df.iterrows()
        ]

        user_point = {
            "x": usage,
            "y": sleep
        }

        classification = classify_academic_impact(int(prediction))
        message = (
            f"Con un uso diario de {usage} horas de redes sociales, {sleep} horas de sueño "
            f"y {conflicts} conflictos relacionados, el modelo predice que tu rendimiento académico "
            f"{'se ve afectado' if prediction == 1 else 'no se ve afectado'} "
            f"con una probabilidad de {(probability * 100):.1f}%."
        )

        recommendations = []
        if prediction == 1:
            recommendations.append("Tu estilo de vida digital podría estar afectando tu rendimiento académico.")
            if usage > 6:
                recommendations.append("Reduce el tiempo en redes sociales, especialmente durante horas de estudio.")
            if sleep < 6:
                recommendations.append("Intenta dormir al menos 7-8 horas por noche para mejorar tu concentración.")
            if conflicts > 2:
                recommendations.append("Busca resolver conflictos relacionados con redes sociales para reducir el estrés.")
        else:
            recommendations.append("Tu estilo de vida digital parece no afectar tu rendimiento académico.")
            recommendations.append("Mantén un equilibrio saludable entre el uso de redes sociales y el descanso.")

        return {
            "affects_academic_performance": int(prediction),
            "academic_impact_classification": message,
            "probability": round(float(probability), 3),
            "user_point": user_point,
            "dataset_points": dataset_points[:100],
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Error en predict_academic_impact: {str(e)}")
        return {
            "affects_academic_performance": 0,
            "academic_impact_classification": "No se pudo predecir el impacto académico.",
            "probability": 0.0,
            "user_point": {"x": 0, "y": 0},
            "dataset_points": [],
            "recommendations": [],
            "error": str(e)
        }
        
	
def academic_performance_risk(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predice el riesgo de que el rendimiento académico se vea afectado por el estilo de vida digital,
    utilizando un modelo de regresión logística basado en uso de redes sociales, horas de sueño y puntaje de salud mental.
    """
    try:
        usage = float(user_data.get("social_media_usage", 0))
        sleep = float(user_data.get("sleep_hours_per_night", 0))
        conflicts = int(user_data.get("conflicts_over_social_media", 0))

        if not 0 <= usage <= 24:
            raise ValueError("El uso de redes sociales debe estar entre 0 y 24 horas.")
        if not 0 <= sleep <= 12:
            raise ValueError("Las horas de sueño deben estar entre 0 y 12 horas.")
        if conflicts < 0:
            raise ValueError("Los conflictos deben ser un número entero no negativo.")

        addicted_score = calculate_addicted_score(usage, conflicts)
        affects_academic = calculate_affects_academic(addicted_score, sleep)
        mental_score = calculate_mental_health_score(usage, sleep, conflicts, addicted_score, affects_academic)

        df = load_dataset()
        df = df[
            (df["Avg_Daily_Usage_Hours"].between(0, 24)) &
            (df["Sleep_Hours_Per_Night"].between(0, 12)) &
            (df["Mental_Health_Score"].notnull()) &
            (df["Mental_Health_Score"].between(0, 10)) &  
            (df["Affects_Academic_Performance"].isin([0, 1]))
        ]

        model, scaler = train_academic_performance_risk_model(df)

        input_df = pd.DataFrame([{
            "Avg_Daily_Usage_Hours": usage,
            "Sleep_Hours_Per_Night": sleep,
            "Mental_Health_Score": mental_score
        }])
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  

        if sleep < 6 and conflicts > 2 and probability < 0.5:
            prediction = 1
            probability = max(probability, 0.6)  

        dataset_stats = {
            "avg_usage_hours": round(float(df["Avg_Daily_Usage_Hours"].mean()), 2),
            "avg_sleep_hours": round(float(df["Sleep_Hours_Per_Night"].mean()), 2),
            "avg_mental_health_score": round(float(df["Mental_Health_Score"].mean()), 2),
            "avg_academic_impact": round(float(df["Affects_Academic_Performance"].mean()), 2)
        }

        graph_data = plot_academic_risk_pie(probability)

        message = (
            f"Con un uso diario de {usage} horas de redes sociales, {sleep} horas de sueño "
            f"y un puntaje de salud mental de {round(mental_score, 2)}, el modelo predice un riesgo "
            f"{'alto' if prediction == 1 else 'bajo'} de que tu rendimiento académico se vea afectado "
            f"con una probabilidad de {(probability * 100):.1f}%."
        )
        if prediction == 1:
            message += " Los factores principales son el bajo tiempo de sueño y los conflictos relacionados con redes sociales."

        recommendations = []
        if prediction == 1:
            recommendations.append("Tu estilo de vida digital podría estar afectando tu rendimiento académico.")
            if usage >= 6:
                recommendations.append("Reduce el tiempo en redes sociales, especialmente durante horas de estudio.")
            if sleep < 6:
                recommendations.append("Intenta dormir al menos 7-8 horas por noche para mejorar tu concentración.")
            if conflicts >= 3:
                recommendations.append("Busca resolver conflictos relacionados con redes sociales para reducir el estrés.")
            if mental_score < 5:  
                recommendations.append("Considera buscar apoyo profesional para mejorar tu salud mental.")
        else:
            recommendations.append("Tu estilo de vida digital parece no afectar tu rendimiento académico.")
            if usage >= 6:
                recommendations.append("Monitorea tu uso de redes sociales para mantener un equilibrio saludable.")
            if sleep < 6:
                recommendations.append("Intenta aumentar tus horas de sueño a 7-8 por noche para optimizar tu rendimiento.")
            recommendations.append("Continúa gestionando el estrés y manteniendo hábitos saludables.")

        return {
            "risk": "Alto" if prediction == 1 else "Bajo",
            "probability": round(float(probability), 4),
            "message": message,
            "dataset_stats": dataset_stats,
            "graph_data": graph_data,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Error en academic_performance_risk: {str(e)}")
        return {
            "risk": "Bajo",
            "probability": 0,
            "message": "No se pudo predecir el riesgo académico debido a un error.",
            "dataset_stats": {},
            "graph_data": {},
            "recommendations": [],
            "error": str(e)
        }
    
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


def social_media_addiction_risk(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predice el riesgo de adicción a redes sociales utilizando un modelo de regresión logística
    basado en uso de redes sociales, horas de sueño, puntaje de salud mental y conflictos.
    """
    try:
        usage = float(user_data.get("social_media_usage", 0))
        sleep = float(user_data.get("sleep_hours_per_night", 0))
        conflicts = int(user_data.get("conflicts_over_social_media", 0))

        if not 0 <= usage <= 24:
            raise ValueError("El uso de redes sociales debe estar entre 0 y 24 horas.")
        if not 0 <= sleep <= 12:
            raise ValueError("Las horas de sueño deben estar entre 0 y 12 horas.")
        if conflicts < 0:
            raise ValueError("Los conflictos deben ser un número entero no negativo.")

        addicted_score = calculate_addicted_score(usage, conflicts)
        affects_academic = calculate_affects_academic(addicted_score, sleep)
        mental_score = calculate_mental_health_score(usage, sleep, conflicts, addicted_score, affects_academic)

        df = load_dataset()
        df = df[
            (df["Avg_Daily_Usage_Hours"].between(0, 24)) &
            (df["Sleep_Hours_Per_Night"].between(0, 12)) &
            (df["Mental_Health_Score"].notnull()) &
            (df["Mental_Health_Score"].between(0, 10)) &
            (df["Addicted_Score"].notnull()) &
            (df["Conflicts_Over_Social_Media"] >= 0)
        ]

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
        probability_high = probabilities[1] 
        if sleep < 6 and conflicts > 2 and probability_high < 0.5:
            prediction = 1
            probability_high = max(probability_high, 0.6)
            probabilities = [1 - probability_high, probability_high]

        graph_data = plot_addiction_risk_bar(probabilities[0], probabilities[1])

        message = (
            f"Con un uso diario de {usage} horas de redes sociales, {sleep} horas de sueño, "
            f"{conflicts} conflictos relacionados y un puntaje de salud mental de {round(mental_score, 2)}, "
            f"el modelo predice un riesgo {'alto' if prediction == 1 else 'bajo'} de adicción a redes sociales "
            f"con una probabilidad de {(probability_high * 100):.1f}%."
        )
        if prediction == 1:
            message += " Los factores principales son el bajo tiempo de sueño y los conflictos relacionados con redes sociales."

        recommendations = []
        if prediction == 1:
            recommendations.append("Tu uso de redes sociales podría indicar un riesgo de adicción.")
            recommendations.append("Limite el tiempo diario en redes sociales usando herramientas de control de tiempo en pantalla.")
            recommendations.append("Busque apoyo profesional, como un psicólogo, para abordar posibles signos de adicción.")
            recommendations.append("Participe en actividades fuera de línea, como deportes o hobbies, para reducir la dependencia de las redes sociales.")
            if usage >= 6:
                recommendations.append("Reduce el tiempo en redes sociales, especialmente en la noche.")
            if sleep < 6:
                recommendations.append("Intenta dormir al menos 7-8 horas por noche para mejorar tu bienestar.")
            if conflicts >= 3:
                recommendations.append("Busca resolver conflictos relacionados con redes sociales para reducir el estrés.")
            if mental_score < 5: 
                recommendations.append("Considera buscar apoyo profesional para mejorar tu salud mental.")
        else:
            recommendations.append("Tu uso de redes sociales parece no indicar un riesgo de adicción.")
            recommendations.append("Mantenga un uso responsable de las redes sociales para evitar el desarrollo de hábitos adictivos.")
            recommendations.append("Monitoree periódicamente su tiempo en redes sociales para mantener un equilibrio saludable.")
            if usage >= 6:
                recommendations.append("Establezca límites claros para el uso de redes sociales, como horarios específicos.")
            if sleep < 6:
                recommendations.append("Intenta aumentar tus horas de sueño a 7-8 por noche para optimizar tu bienestar.")

        return {
            "risk": "Alto" if prediction == 1 else "Bajo",
            "probabilities": {
                "No adicción": round(probabilities[0], 3),
                "Adicción": round(probabilities[1], 3)
            },
            "message": message,
            "graph_data": graph_data,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Error en social_media_addiction_risk: {str(e)}")
        return {
            "risk": "Desconocido",
            "probabilities": {},
            "message": "No se pudo predecir el riesgo de adicción debido a un error.",
            "graph_data": {},
            "recommendations": [],
            "error": str(e)
        }

def visualize_decision_tree(target_column: str, user_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Genera una visualización del árbol de decisión y recomendaciones personalizadas basadas en los datos del usuario.

    Args:
        target_column: Columna objetivo para la clasificación (por ejemplo, 'Addicted_Score').
        user_data: Diccionario con datos del usuario (Avg_Daily_Usage_Hours, Sleep_Hours_Per_Night, Mental_Health_Score,
                   Addicted_Score, Conflicts_Over_Social_Media).

    Returns:
        Diccionario con el texto del árbol, predicción del usuario, mensaje y recomendaciones.
    """
    try:
        df = load_dataset()

        model, features = train_decision_tree_model(df, target_column)

        tree_text = export_text(model, feature_names=features)

        response = {
            "tree_text": tree_text,
            "target": target_column,
            "label": f"Análisis de tus hábitos con base en {target_column}",
            "recommendations": [],
            "severity": "desconocido",
            "prediction": None
        }

        if not user_data:
            response["label"] = f"Análisis de hábitos para {target_column}"
            return response

        usage = float(user_data.get("Avg_Daily_Usage_Hours", 0))
        sleep = float(user_data.get("Sleep_Hours_Per_Night", 0))
        mental_health = float(user_data.get("Mental_Health_Score", 5))
        addicted_score = float(user_data.get("Addicted_Score", 0))
        conflicts = float(user_data.get("Conflicts_Over_Social_Media", 0))

        if usage < 0 or usage > 24:
            raise ValueError("El uso de redes sociales debe estar entre 0 y 24 horas.")
        if sleep < 0 or sleep > 12:
            raise ValueError("Las horas de sueño deben estar entre 0 y 12 horas.")
        if mental_health < 0 or mental_health > 10:
            raise ValueError("El puntaje de salud mental debe estar entre 0 y 10.")
        if addicted_score < 0:
            raise ValueError("El puntaje de adicción debe ser no negativo.")
        if conflicts < 0:
            raise ValueError("Los conflictos deben ser un número no negativo.")

        user_values = pd.DataFrame(
            [[usage, sleep, mental_health, addicted_score, conflicts]],
            columns=features
        )
        prediction = model.predict(user_values)[0]
        logger.debug(f"Predicción para el usuario: {prediction}, tipo: {type(prediction)}")

        if target_column == "Addicted_Score":
            prediction_float = float(prediction)  
            if prediction_float >= 7 or usage > 9 or (usage > 4 and (sleep < 5 or conflicts >= 4 or mental_health < 3)):
                severity = "alto"
                message = (
                    f"Basado en tus hábitos, tienes un alto riesgo de adicción a las redes sociales. "
                    f"Pasas unas {usage:.1f} horas al día en redes, duermes unas {sleep:.1f} horas por noche "
                    f"y tu salud mental está en un nivel de {mental_health:.1f}/10."
                )
            elif prediction_float >= 4 or usage > 4 or (usage >= 2 and (sleep < 6 or conflicts >= 2 or mental_health < 4)):
                severity = "moderado"
                message = (
                    f"Basado en tus hábitos, tienes un riesgo moderado de adicción a las redes sociales. "
                    f"Pasas unas {usage:.1f} horas al día en redes, duermes unas {sleep:.1f} horas por noche "
                    f"y tu salud mental está en un nivel de {mental_health:.1f}/10."
                )
            else:
                severity = "bajo"
                message = (
                    f"Basado en tus hábitos, tienes un bajo riesgo de adicción a las redes sociales. "
                    f"Pasas unas {usage:.1f} horas al día en redes, duermes unas {sleep:.1f} horas por noche "
                    f"y tu salud mental está en un nivel de {mental_health:.1f}/10."
                )
        else:
            severity = str(prediction).lower() if isinstance(prediction, str) else "desconocido"
            message = (
                f"Tu nivel de {target_column} es {prediction}. "
                f"Pasas unas {usage:.1f} horas al día en redes, duermes unas {sleep:.1f} horas por noche "
                f"y tu salud mental está en un nivel de {mental_health:.1f}/10."
            )

        recommendations = []
        if severity == "alto":
            recommendations.extend([
                "Tu tiempo en redes sociales es alto y puede afectar tu bienestar. Intenta reducirlo a menos de 4 horas al día.",
                "Considera hablar con un profesional, como un psicólogo, para manejar el impacto de las redes en tu vida.",
                "Dedica más tiempo a actividades sin pantallas, como salir a caminar, leer o pasar tiempo con amigos.",
                "Establece un horario para dormir mejor, evitando usar el teléfono al menos una hora antes de acostarte."
            ])
        elif severity == "moderado":
            recommendations.extend([
                "Tu uso de redes sociales está en un nivel intermedio. Prueba limitarlo a 2-4 horas al día para un mejor equilibrio.",
                "Asegúrate de dormir entre 7 y 8 horas por noche para mejorar tu energía y salud mental.",
                "Si las redes sociales causan discusiones, intenta resolverlas hablando con las personas involucradas."
            ])
        else:
            recommendations.extend([
                "¡Bien hecho! Tu uso de redes sociales parece equilibrado. Sigue manteniendo estos hábitos saludables.",
                "Continúa cuidando tu sueño y tu salud mental para mantenerte en buen estado."
            ])

        if usage > 9:
            recommendations.append("Pasas mucho tiempo en redes sociales. Usa aplicaciones de control de tiempo para establecer límites diarios.")
        elif usage > 4:
            recommendations.append("Reducir un poco tu tiempo en redes sociales te dará más espacio para otras actividades.")
        if sleep < 6:
            recommendations.append("Dormir menos de 6 horas por noche puede afectar tu salud. Intenta descansar entre 7 y 8 horas.")
        if mental_health < 4:
            recommendations.append("Tu salud mental parece estar baja. Considera buscar apoyo profesional o hablar con alguien de confianza.")
        if conflicts >= 3:
            recommendations.append("Los conflictos frecuentes por redes sociales pueden ser estresantes. Habla con alguien para encontrar soluciones.")
        elif conflicts >= 1:
            recommendations.append("Si las redes sociales generan discusiones, establece reglas claras para su uso con tus seres queridos.")
        if addicted_score >= 7:
            recommendations.append("Tu puntaje de adicción es alto. Busca apoyo para reducir tu dependencia de las redes sociales.")

        response.update({
            "label": message,
            "recommendations": recommendations,
            "severity": severity,
            "prediction": float(prediction) if isinstance(prediction, (np.integer, np.floating)) else str(prediction)
        })

        return response

    except Exception as e:
        logger.error(f"Error en visualize_decision_tree: {str(e)}")
        return {
            "tree_text": "",
            "target": target_column,
            "label": "No pudimos analizar tus datos. Por favor, verifica la información proporcionada.",
            "recommendations": [],
            "severity": "desconocido",
            "prediction": None,
            "error": str(e)
        }


def run_kmeans_clustering(user_data: Dict[str, Any], n_clusters: int = 3) -> Dict[str, Any]:
    """
    Realiza clustering K-Means con los datos del usuario y devuelve resultados accesibles.

    Args:
        user_data: Diccionario con datos del usuario (age, social_media_usage, sleep_hours_per_night, conflicts_over_social_media).
        n_clusters: Número de grupos para K-Means (por defecto 3).

    Returns:
        Diccionario con resultados del clustering, incluyendo puntos, punto del usuario, estadísticas y recomendaciones.
    """
    try:
        age = float(user_data.get("age", 0))
        usage = float(user_data.get("social_media_usage", 0))
        sleep = float(user_data.get("sleep_hours_per_night", 0))
        conflicts = float(user_data.get("conflicts_over_social_media", 0))

        if age <= 0:
            raise ValueError("La edad debe ser mayor que 0.")
        if not 0 <= usage <= 24:
            raise ValueError("El uso de redes sociales debe estar entre 0 y 24 horas.")
        if not 0 <= sleep <= 12:
            raise ValueError("Las horas de sueño deben estar entre 0 y 12 horas.")
        if conflicts < 0:
            raise ValueError("Los conflictos deben ser un número no negativo.")

        df = load_dataset()
        features = [
            "Age",
            "Avg_Daily_Usage_Hours",
            "Sleep_Hours_Per_Night",
            "Conflicts_Over_Social_Media"
        ]
        df = df[features].dropna()

        model, _, X_scaled, scaler = train_kmeans_model(df, n_clusters)

        df_plot = pd.DataFrame(X_scaled[:, :2], columns=["x", "y"])
        df_plot["cluster"] = model.labels_
        points = df_plot.to_dict(orient="records")

        user_values = [[age, usage, sleep, conflicts]]
        user_scaled = scaler.transform(user_values)
        user_cluster = int(model.predict(user_scaled)[0])

        user_point = {
            "x": float(user_scaled[0][0]),
            "y": float(user_scaled[0][1]),
            "cluster": user_cluster,
            "label": "Tu Predicción"
        }

        df["cluster"] = model.labels_
        cluster_stats = df.groupby("cluster")[features].mean().to_dict(orient="index")
        user_cluster_stats = cluster_stats.get(user_cluster, {})
        usage_avg = user_cluster_stats.get("Avg_Daily_Usage_Hours", 0)
        sleep_avg = user_cluster_stats.get("Sleep_Hours_Per_Night", 0)
        conflicts_avg = user_cluster_stats.get("Conflicts_Over_Social_Media", 0)

        if usage > 9 or (usage > 4 and (sleep < 5 or conflicts >= 4)):
            severity = "alto"
            message = (
                f"Tu comportamiento se asemeja a un grupo con alto riesgo. "
                f"Las personas de este grupo suelen pasar unas {usage_avg:.1f} horas al día en redes sociales, "
                f"dormir unas {sleep_avg:.1f} horas por noche y tener conflictos frecuentes relacionados con redes sociales."
            )
        elif usage > 4 or (usage >= 2 and (sleep < 6 or conflicts >= 2)):
            severity = "moderado"
            message = (
                f"Tu comportamiento se asemeja a un grupo con riesgo moderado. "
                f"Las personas de este grupo pasan unas {usage_avg:.1f} horas al día en redes sociales, "
                f"duermen unas {sleep_avg:.1f} horas por noche y a veces tienen conflictos por redes sociales."
            )
        else:
            severity = "bajo"
            message = (
                f"Tu comportamiento se asemeja a un grupo con bajo riesgo. "
                f"Las personas de este grupo pasan unas {usage_avg:.1f} horas al día en redes sociales, "
                f"duermen unas {sleep_avg:.1f} horas por noche y tienen pocos conflictos relacionados."
            )

        recommendations = []
        if severity == "alto":
            recommendations.extend([
                "Pasas mucho tiempo en redes sociales, lo que puede afectar tu salud y relaciones. Intenta reducir tu uso a menos de 4 horas al día.",
                "Considera buscar apoyo profesional, como un psicólogo, para manejar el impacto de las redes sociales en tu vida.",
                "Dedica más tiempo a actividades fuera de línea, como salir con amigos, hacer ejercicio o leer.",
                "Crea un horario para dormir mejor, evitando las redes sociales al menos una hora antes de acostarte."
            ])
        elif severity == "moderado":
            recommendations.extend([
                "Tu uso de redes sociales está en un nivel intermedio. Intenta limitarlo a 2-4 horas al día para mantener un buen equilibrio.",
                "Asegúrate de dormir entre 7 y 8 horas por noche para mejorar tu energía y bienestar.",
                "Si las redes sociales causan tensiones, habla con las personas involucradas para resolver cualquier problema."
            ])
        else:
            recommendations.extend([
                "¡Excelente! Tu uso de redes sociales es equilibrado. Sigue manteniendo estos hábitos saludables.",
                "Continúa priorizando un buen descanso y relaciones positivas en tu vida diaria."
            ])

        if usage > 9:
            recommendations.append("Tu uso de redes sociales es muy alto. Prueba establecer un límite diario usando alarmas o aplicaciones de control de tiempo.")
        elif usage > 4:
            recommendations.append("Reducir un poco tu tiempo en redes sociales puede ayudarte a tener más tiempo para otras actividades.")
        if sleep < 6:
            recommendations.append("Dormir menos de 6 horas por noche puede afectar tu salud. Intenta acostarte más temprano y evitar pantallas antes de dormir.")
        if conflicts >= 3:
            recommendations.append("Los conflictos frecuentes por redes sociales pueden ser estresantes. Considera hablar con alguien de confianza o un profesional para manejarlos.")
        elif conflicts >= 1:
            recommendations.append("Si las redes sociales generan discusiones, intenta establecer reglas claras para su uso con tus seres queridos.")

        return {
            "clusters": n_clusters,
            "features_used": ["Edad", "Horas de Uso de Redes Sociales"],
            "points": points,
            "user_point": user_point,
            "label": message,
            "recommendations": recommendations,
            "severity": severity,
            "cluster_stats": {k: {fk: float(fv) for fk, fv in v.items()} for k, v in cluster_stats.items()}
        }

    except Exception as e:
        logger.error(f"Error en run_kmeans_clustering: {str(e)}")
        return {
            "clusters": n_clusters,
            "features_used": [],
            "points": [],
            "user_point": {},
            "label": "No pudimos analizar tus datos. Por favor, verifica la información proporcionada.",
            "recommendations": [],
            "severity": "desconocido",
            "cluster_stats": {},
            "error": str(e)
        }