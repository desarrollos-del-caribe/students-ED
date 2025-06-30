from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    try:
        conn = Config.get_connection()
        query = """
            SELECT sm.age, sm.gender_id, sm.academic_level_id, sm.country_id, sm.avg_daily_used_hours, 
                   sm.social_network_id, sm.affects_academic_performance, sm.sleep_hours_per_night, 
                   sm.mental_health_score, sm.relationship_status_id, sm.conflicts_over_social_media, 
                   sm.addicted_score, sn.name_social_network, c.name_country
            FROM Tbl_Students_Model sm
            JOIN Tbl_Socials_Networks sn ON sm.social_network_id = sn.id
            JOIN Tbl_Countries c ON sm.country_id = c.id
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error cargando datos: {str(e)}")
        raise e

def social_media_addiction_risk(usage_hours, addicted_score, mental_health_score, conflicts_score):
    try:
        df = load_data()
        df = df.fillna(0)
        
        def clasificar_riesgo(row):
            if row['avg_daily_used_hours'] > 5 and row['addicted_score'] >= 7 and row['conflicts_over_social_media'] >= 2:
                return "Alto"
            elif row['addicted_score'] >= 5 or row['mental_health_score'] <= 5:
                return "Medio"
            else:
                return "Bajo"
        
        df['Riesgo_Adiccion'] = df.apply(clasificar_riesgo, axis=1)
        X = df[['avg_daily_used_hours', 'addicted_score', 'mental_health_score', 'conflicts_over_social_media']]
        y = df['Riesgo_Adiccion']
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        pred = model.predict(pd.DataFrame([{
            'avg_daily_used_hours': usage_hours, 
            'addicted_score': addicted_score, 
            'mental_health_score': mental_health_score, 
            'conflicts_over_social_media': conflicts_score
        }]))[0]
        return pred
    except Exception as e:
        logger.error(f"Error en predicción de riesgo de adicción: {str(e)}")
        return "Error"

def academic_performance_risk(usage_hours, sleep_hours, mental_health_score):
    try:
        df = load_data()
        df = df.fillna(0)
        X = df[['avg_daily_used_hours', 'sleep_hours_per_night', 'mental_health_score']]
        y = df['affects_academic_performance']
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        pred_prob = model.predict_proba(pd.DataFrame([{
            'avg_daily_used_hours': usage_hours, 
            'sleep_hours_per_night': sleep_hours, 
            'mental_health_score': mental_health_score
        }]))[0][1]
        return "Alto" if pred_prob > 0.7 else "Medio" if pred_prob > 0.4 else "Bajo"
    except Exception as e:
        logger.error(f"Error en predicción de impacto académico: {str(e)}")
        return "Error"

def sleep_prediction(usage_hours, age, mental_health_score):
    try:
        df = load_data()
        df = df.fillna(0)
        X = df[['avg_daily_used_hours', 'age', 'mental_health_score']]
        y = df['sleep_hours_per_night']
        model = LinearRegression()
        model.fit(X, y)
        pred_sleep = model.predict(pd.DataFrame([{
            'avg_daily_used_hours': usage_hours, 
            'age': age, 
            'mental_health_score': mental_health_score
        }]))[0]
        return max(0, min(12, pred_sleep))
    except Exception as e:
        logger.error(f"Error en predicción de horas de sueño: {str(e)}")
        return 0

def get_platform_distribution():
    try:
        df = load_data()
        platform_counts = df['name_social_network'].value_counts().to_dict()
        return {
            'labels': list(platform_counts.keys()),
            'data': list(platform_counts.values())
        }
    except Exception as e:
        logger.error(f"Error en distribución de plataformas: {str(e)}")
        return {'labels': [], 'data': []}

def get_country_impact():
    try:
        df = load_data()
        country_impact = df.groupby('name_country')['affects_academic_performance'].mean().to_dict()
        return {
            'labels': list(country_impact.keys()),
            'data': list(country_impact.values())
        }
    except Exception as e:
        logger.error(f"Error en impacto por país: {str(e)}")
        return {'labels': [], 'data': []}

def get_usage_mental_health_correlation():
    try:
        df = load_data()
        return {
            'usage_hours': df['avg_daily_used_hours'].tolist(),
            'mental_health': df['mental_health_score'].tolist()
        }
    except Exception as e:
        logger.error(f"Error en correlación uso-salud mental: {str(e)}")
        return {'usage_hours': [], 'mental_health': []}
