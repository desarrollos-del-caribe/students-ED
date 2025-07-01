import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from config import Config
import logging

logger = logging.getLogger(__name__)

# def load_dataDefault():
#     try:
#         conn = Config.get_connection()
#         query = """
#         SELECT FROM * Tbl_Students_Model
        
#         "
        

def load_data(historyModel=None):
    """Carga datos de Tbl_Students_Model, filtrando por ID si se proporciona."""
    try:
        conn = Config.get_connection()

        if historyModel is not None:
            query = f"""
                SELECT *
                FROM Tbl_Students_Model
                WHERE history_models_import = 1 AND id_history_model = {historyModel}
            """
        else:
            query = """
                SELECT *
                FROM Tbl_Students_Model
                WHERE history_models_import = 1
            """

        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame()


def social_media_addiction_risk(usage_hours, addicted_score, mental_health_score, conflicts_score, historyModel=None):
    """Predice riesgo de adicción usando DecisionTreeClassifier."""
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

def sleep_prediction(usage_hours, age, mental_health_score, historyModel=None):
    """Predice horas de sueño usando LinearRegression."""
    try:
        df = load_data(historyModel)
        df = clean_data(df)

        features = ['age', 'avg_daily_used_hours', 'mental_health_score', 'sleep_hours_per_night']
        if not all(col in df.columns for col in features):
            logger.error("Columnas necesarias no encontradas")
            return 0

        X = df[['age', 'avg_daily_used_hours', 'mental_health_score']]
        y = df['sleep_hours_per_night']

        model = LinearRegression()
        model.fit(X, y)

        entrada = pd.DataFrame([{
            'age': age,
            'avg_daily_used_hours': usage_hours,
            'mental_health_score': mental_health_score
        }])

        pred = model.predict(entrada)[0]
        return round(pred, 2)
    except Exception as e:
        logger.error(f"Error en sleep_prediction: {str(e)}")
        return 0

def academic_performance_risk(usage_hours, sleep_hours, mental_health_score, historyModel=None):
    """Predice riesgo académico usando LogisticRegression."""
    try:
        df = load_data(historyModel)
        df = clean_data(df)

        required_cols = ['avg_daily_used_hours', 'sleep_hours_per_night', 'mental_health_score', 'affects_academic_performance']
        if not all(col in df.columns for col in required_cols):
            logger.error("Columnas requeridas no encontradas")
            return "Bajo"

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

def get_platform_distribution(historyModel=None):
    """Obtiene distribución de plataformas."""
    try:
        df = load_data(historyModel)
        df = clean_data(df)

        if 'social_network_id' not in df.columns:
            logger.error("Columna 'social_network_id' no encontrada")
            return {"labels": ["Plataforma 1", "Plataforma 2", "Plataforma 3", "Plataforma 4"], "data": [1, 1, 1, 1]}

        conn = Config.get_connection()
        cursor = conn.cursor(as_dict=True)
        cursor.execute("SELECT id, name FROM Tbl_Social_Network")
        platform_map = {row['id']: row['name'] for row in cursor.fetchall()}
        conn.close()

        platform_counts = df['social_network_id'].value_counts(normalize=True) * 100
        labels = [platform_map.get(id, f"Plataforma {id}") for id in platform_counts.index]
        data = platform_counts.values.tolist()

        return {"labels": labels, "data": [round(x, 2) for x in data]}
    except Exception as e:
        logger.error(f"Error en get_platform_distribution: {str(e)}")
        return {"labels": ["Plataforma 1", "Plataforma 2", "Plataforma 3", "Plataforma 4"], "data": [1, 1, 1, 1]}