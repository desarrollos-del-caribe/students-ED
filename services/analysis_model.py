import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from config import Config
import logging

logger = logging.getLogger(__name__)

def clean_data(df):
    """Limpia los datos manejando valores nulos y tipos de datos."""
    df = df.copy()
    numeric_cols = ['avg_daily_used_hours', 'addicted_score', 'mental_health_score', 
                    'conflicts_over_social_media', 'sleep_hours_per_night', 'age']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_value = df[col].mean() if not df[col].isna().all() else 0
            df[col] = df[col].fillna(mean_value)
    return df

def load_data(historyModel=None):
    """Carga datos de Tbl_Students_Model con unión a Tbl_Countries."""
    try:
        conn = Config.get_connection()
        if conn is None:
            raise Exception("No se pudo establecer la conexión a la base de datos")
        
        if historyModel is not None:
            query = """
                SELECT s.*, c.name_country AS country
                FROM Tbl_Students_Model s
                LEFT JOIN Tbl_Countries c ON s.country_id = c.id
                WHERE s.history_models_import_id = %s
            """
            params = (historyModel,)
        else:
            query = """
                SELECT s.*, c.name_country AS country
                FROM Tbl_Students_Model s
                LEFT JOIN Tbl_Countries c ON s.country_id = c.id
            """
            params = ()

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error cargando datos: {str(e)}")
        if 'conn' in locals() and conn is not None:
            conn.close()
        return pd.DataFrame()

def social_media_addiction_risk(usage_hours, addicted_score, mental_health_score, conflicts_score, historyModel=None):
    """Predice riesgo de adicción usando DecisionTreeClassifier basado en el historial."""
    try:
        df = load_data(historyModel)
        df = clean_data(df)

        required_cols = ['avg_daily_used_hours', 'addicted_score', 'mental_health_score', 'conflicts_over_social_media']
        if not all(col in df.columns for col in required_cols):
            logger.error("Columnas requeridas no encontradas")
            return "Bajo"

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

        entrada = pd.DataFrame([{
            'avg_daily_used_hours': usage_hours,
            'addicted_score': addicted_score,
            'mental_health_score': mental_health_score,
            'conflicts_over_social_media': conflicts_score
        }])

        pred = model.predict(entrada)[0]
        return pred
    except Exception as e:
        logger.error(f"Error en social_media_addiction_risk: {str(e)}")
        return "Bajo"

def academic_performance_risk(usage_hours, sleep_hours, mental_health_score, historyModel=None):
    """Predice riesgo académico usando LogisticRegression basado en el historial."""
    try:
        df = load_data(historyModel)
        df = clean_data(df)

        required_cols = ['avg_daily_used_hours', 'sleep_hours_per_night', 'mental_health_score', 'affects_academic_performance']
        if not all(col in df.columns for col in required_cols):
            logger.error("Columnas requeridas no encontradas")
            return {"risk": "Bajo", "probability": 0}

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
        return {"risk": "Alto" if pred == 1 else "Bajo", "probability": round(prob, 4)}
    except Exception as e:
        logger.error(f"Error en academic_performance_risk: {str(e)}")
        return {"risk": "Bajo", "probability": 0}

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
        
        # Comparación con el promedio del historial
        avg_addicted_score = df['addicted_score'].mean() if not df['addicted_score'].isna().all() else 0
        avg_academic_risk = df['affects_academic_performance'].mean() if not df['affects_academic_performance'].isna().all() else 0
        return {
            "id": student_id,
            "addiction_risk": addiction_pred,
            "addiction_score_vs_avg": round(student_data['addicted_score'].iloc[0] - avg_addicted_score, 2),
            "academic_risk": academic_pred['risk'],
            "academic_risk_probability": academic_pred['probability'],
            "academic_risk_vs_avg": round(student_data['affects_academic_performance'].iloc[0] - avg_academic_risk, 2) if 'affects_academic_performance' in student_data.columns else 0
        }
    except Exception as e:
        logger.error(f"Error en student_performance_prediction: {str(e)}")
        return {"error": str(e)}

def addiction_by_country(historyModel=None):
    """Calcula el riesgo promedio de adicción por país basado en el historial."""
    try:
        df = load_data(historyModel)
        df = clean_data(df)

        if 'country' not in df.columns or 'addicted_score' not in df.columns:
            logger.error("Columnas 'country' o 'addicted_score' no encontradas")
            return {"error": "Columnas requeridas no encontradas"}

        country_risk = df.groupby('country')['addicted_score'].mean().to_dict()
        return {country: round(score, 2) for country, score in country_risk.items()}
    except Exception as e:
        logger.error(f"Error en addiction_by_country: {str(e)}")
        return {"error": str(e)}