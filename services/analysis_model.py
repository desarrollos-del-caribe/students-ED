# Predicciones

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from excel_service import load_dataset
from ml_service import train_social_media_addiction_model
from utils.helpers import generate_risk_recommendations # Para los mensajes o consejos

import logging
logger = logging.getLogger(__name__)

def predict_social_media_addiction_risk(usage_hours, addicted_score, mental_health_score, conflicts_score):
    try:
        df = load_dataset()
        model = train_social_media_addiction_model(df)

        input_df = pd.DataFrame([{
            'avg_daily_used_hours': usage_hours,
            'addicted_score': addicted_score,
            'mental_health_score': mental_health_score,
            'conflicts_over_social_media': conflicts_score
        }])

        pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_
        probabilities = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}

        dataset_stats = {
            "avg_usage_hours": round(float(df['avg_daily_used_hours'].mean()), 2),
            "avg_addicted_score": round(float(df['addicted_score'].mean()), 2),
            "avg_mental_health_score": round(float(df['mental_health_score'].mean()), 2),
            "avg_conflicts_score": round(float(df['conflicts_over_social_media'].mean()), 2)
        }

        recommendations = generate_risk_recommendations(
            risk_score=0,  # Aquí puedes definir una fórmula real
            risk_factors=[pred]
        )

        return {
            "risk": pred,
            "probabilities": probabilities,
            "dataset_stats": dataset_stats,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Error en predict_social_media_addiction_risk: {str(e)}")
        return {
            "risk": "Bajo",
            "probabilities": {"Bajo": 1.0, "Medio": 0.0, "Alto": 0.0},
            "dataset_stats": {},
            "recommendations": []
        }

def academic_performance_risk(usage_hours, sleep_hours, mental_health_score, historyModel=None):
    """Predice riesgo académico usando LogisticRegression basado en el historial."""
    try:
        df = load_data(historyModel)
        df = clean_data(df)

        required_cols = ['avg_daily_used_hours', 'sleep_hours_per_night', 'mental_health_score', 'affects_academic_performance']
        if not all(col in df.columns for col in required_cols):
            logger.error("Columnas requeridas no encontradas")
            return {"risk": "Bajo", "probability": 0, "dataset_stats": {}}

        X = df[['avg_daily_used_hours', 'sleep_hours_per_night', 'mental_health_score']]
        y = df['affects_academic_performance']

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        entrada = pd.DataFrame([{
            'avg_daily_used_hours': usage_hours,
            'sleep_hours_per_night': sleep_hours,
            'mental_health_score': mental_health_score
        }])

        prob = model.predict_proba(entrada)[0][1]
        pred = model.predict(entrada)[0]

        dataset_stats = {
            "avg_usage_hours": round(float(df['avg_daily_used_hours'].mean()), 2),
            "avg_sleep_hours": round(float(df['sleep_hours_per_night'].mean()), 2),
            "avg_mental_health_score": round(float(df['mental_health_score'].mean()), 2),
            "avg_academic_impact": round(float(df['affects_academic_performance'].mean()), 2)
        }
        return {
            "risk": "Alto" if pred == 1 else "Bajo",
            "probability": round(prob, 4),
            "dataset_stats": dataset_stats
        }
    except Exception as e:
        logger.error(f"Error en academic_performance_risk: {str(e)}")
        return {"risk": "Bajo", "probability": 0, "dataset_stats": {}}

def student_performance_prediction(student_id, historyModel=None):
    """Predice el rendimiento académico y riesgo de adicción para un estudiante específico."""
    try:
        df = load_data(historyModel)
        df = clean_data(df)

        if 'id' not in df.columns:
            logger.error("Columna 'id' no encontrada")
            return {"error": "Columna 'id' no encontrada"}
        
        student_data = df[df['id'] == student_id]
        if student_data.empty:
            logger.error(f"Estudiante con ID {student_id} no encontrado")
            return {"error": f"Estudiante con ID {student_id} no encontrado"}
        
        required_cols = ['avg_daily_used_hours', 'addicted_score', 'mental_health_score', 
                        'conflicts_over_social_media', 'sleep_hours_per_night']
        if not all(col in df.columns for col in required_cols):
            logger.error("Columnas requeridas no encontradas")
            return {"error": "Columnas requeridas no encontradas"}
        
        addiction_pred = social_media_addiction_risk(
            student_data['avg_daily_used_hours'].iloc[0],
            student_data['addicted_score'].iloc[0],
            student_data['mental_health_score'].iloc[0],
            student_data['conflicts_over_social_media'].iloc[0],
            historyModel
        )
        academic_pred = academic_performance_risk(
            student_data['avg_daily_used_hours'].iloc[0],
            student_data['sleep_hours_per_night'].iloc[0],
            student_data['mental_health_score'].iloc[0],
            historyModel
        )
        dataset_stats = {
            "avg_addicted_score": round(float(df['addicted_score'].mean()), 2),
            "avg_academic_impact": round(float(df['affects_academic_performance'].mean()), 2),
            "student_addicted_score": round(float(student_data['addicted_score'].iloc[0]), 2),
            "student_academic_impact": round(float(student_data['affects_academic_performance'].iloc[0]), 2) if 'affects_academic_performance' in student_data.columns else 0
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

def addiction_by_country(historyModel=None, min_students=5):
    """Calcula el riesgo promedio de adicción por país basado en el historial."""
    try:
        df = load_data(historyModel)
        df = clean_data(df)

        if df.empty:
            logger.error("No se encontraron datos para el historial especificado")
            return {"error": "No se encontraron datos para el historial especificado"}

        if 'country' not in df.columns or 'addicted_score' not in df.columns:
            logger.error("Columnas 'country' o 'addicted_score' no encontradas")
            return {"error": "Columnas requeridas no encontradas"}
        
        # Calcular promedio y conteo por país
        country_stats = df.groupby('country').agg({
            'addicted_score': ['mean', 'count'],
            'avg_daily_used_hours': 'mean'
        }).reset_index()
        
        country_stats.columns = ['country', 'addicted_score_mean', 'addicted_score_count', 'avg_daily_used_hours_mean']
        
        # Filtrar países con menos de min_students
        country_stats = country_stats[country_stats['addicted_score_count'] >= min_students]
        
        result = {
            "countries": [],
            "avg_addicted_scores": [],
            "student_counts": [],
            "avg_usage_hours": []
        }
        
        for _, row in country_stats.iterrows():
            result["countries"].append(row['country'])
            result["avg_addicted_scores"].append(round(float(row['addicted_score_mean']), 2))
            result["student_counts"].append(int(row['addicted_score_count']))
            result["avg_usage_hours"].append(round(float(row['avg_daily_used_hours_mean']), 2))

        # Estadísticas generales
        dataset_stats = {
            "total_students": len(df),
            "avg_addicted_score": round(float(df['addicted_score'].mean()), 2) if not df['addicted_score'].isna().all() else 0,
            "avg_usage_hours": round(float(df['avg_daily_used_hours'].mean()), 2) if not df['avg_daily_used_hours'].isna().all() else 0
        }

        return {
            "country_data": result,
            "dataset_stats": dataset_stats
        }
    except Exception as e:
        logger.error(f"Error en addiction_by_country: {str(e)}")
        return {"error": str(e)}