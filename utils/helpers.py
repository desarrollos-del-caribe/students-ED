import os
import pandas as pd
from datetime import datetime
import json
from typing import Dict, Any, List
import logging
import os
import time
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_DIR = os.path.join(os.path.dirname(__file__), '../static/graphs')

def save_plot_image_with_timestamp(name_prefix):
    """
    Guarda el gráfico actual como imagen con nombre único basado en timestamp.
    """
    timestamp = int(time.time())
    filename = f"{name_prefix}_{timestamp}.png"
    filepath = os.path.join(GRAPH_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return f"/static/graphs/{filename}"

#Eliminar PNGs de graficas de 5 min de antiguedad
def clean_old_graphs(directory, max_age_seconds=300):
    """
    Elimina archivos PNG en la carpeta que sean más viejos que max_age_seconds (default: 5 minutos).
    """
    now = time.time()
    deleted = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Solo archivos .png
        if filename.endswith(".png"):
            file_age = now - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                os.remove(filepath)
                deleted.append(filename)

    return deleted

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convertir valor a float de forma segura"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Convertir valor a int de forma segura"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_bool(value: Any, default: bool = False) -> bool:
    """Convertir valor a bool de forma segura"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ['true', '1', 'yes', 'on', 'y']
    if isinstance(value, (int, float)):
        return bool(value)
    return default

def calculate_correlation(data: pd.DataFrame, col1: str, col2: str) -> float:
    """Calcular correlación entre dos columnas"""
    try:
        if col1 in data.columns and col2 in data.columns:
            return float(data[col1].corr(data[col2]))
    except:
        pass
    return 0.0

def calculate_addicted_score(usage_hours, conflicts):
    """
    Calcula el nivel de adicción a redes sociales.
    Usa una fórmula simple basada en uso diario y conflictos.

    Retorna un puntaje entre 0 y 10.
    """
    score = usage_hours * 1.5 + conflicts * 1.2
    return min(10, round(score, 2))

def calculate_affects_academic(addicted_score, sleep_hours):
    """
    Evalúa si el usuario presenta afectación en su rendimiento académico.
    Si tiene alta adicción (>6) o duerme poco (<5h), se considera afectado.

    Retorna 1 (sí afecta) o 0 (no afecta).
    """
    return 1 if addicted_score > 6 or sleep_hours < 5 else 0

def calculate_mental_health_score(usage_hours, sleep_hours, conflicts, addicted_score, affects_academic):
    """
    Calcula un puntaje general de salud mental (escala de 1 a 10).
    Penaliza el uso excesivo, conflictos y falta de sueño. Premia buenos hábitos.

    Retorna el puntaje como número entre 1 y 10.
    """
    # Puntaje base sobre 100
    score = 100
    score -= usage_hours * 5
    score += sleep_hours * 4
    score -= conflicts * 3
    score -= addicted_score * 3
    if affects_academic:
        score -= 10

    score = max(0, min(100, round(score, 2)))

    # Escalar el puntaje de 0-100 a 1-10
    mental_score = round((score / 100) * 9 + 1, 2)
    return mental_score

def classify_mental_health(score):
    """
    Clasifica la salud mental en base al puntaje (1 a 10).
    Retorna una descripción textual amigable y útil.
    """
    msg = f"Tu puntaje de salud mental es {score}. "
    if score >= 8:
        return msg + "Tienes una excelente salud mental. Continúa con tus hábitos positivos."
    elif score >= 6:
        return msg + "Tu salud mental es buena, aunque podrías mejorar algunos aspectos."
    elif score >= 4:
        return msg + "Tu salud mental es moderada. Considera reducir el uso de redes sociales y mejorar tu descanso."
    else:
        return msg + "Tu salud mental parece baja. Sería bueno hablar con alguien y cuidar tu bienestar."

def classify_sleep_quality(hours: float) -> str:
    """Clasifica las horas de sueño con mensajes descriptivos"""
    msg = f"Registraste {hours} horas de sueño por noche. "
    if hours >= 8:
        return msg + "Tus horas de sueño son excelentes. Mantén ese buen hábito para cuidar tu salud física y mental."
    elif hours >= 6.5:
        return msg + "Duermes una cantidad aceptable, pero podrías beneficiarte de un poco más de descanso cada noche."
    elif hours >= 5:
        return msg +"Tus horas de sueño son bajas. Trata de dormir más para mejorar tu concentración y bienestar general."
    else:
        return "Tus horas de sueño son muy bajas. Esto puede afectar seriamente tu salud y desempeño académico. Considera establecer una rutina de descanso más saludable."

def classify_platform_risk(platform: str) -> str:
    """Clasifica el riesgo potencial de la plataforma más usada."""
    risks = {
        'Instagram': "Es una plataforma visual que puede fomentar la comparación social. Úsala con moderación.",
        'TikTok': "Contenido muy rápido y adictivo. Puede afectar tu atención si se usa en exceso.",
        'Facebook': "Red tradicional. Su impacto depende del tipo de contenido que consumes.",
        'YouTube': "Puede ser educativa o adictiva, según el uso que le des.",
        'Twitter': "Puede generar ansiedad por sobreexposición a noticias o debates.",
        'LinkedIn': "Enfocada en lo profesional. Suele tener bajo impacto negativo.",
        'WhatsApp': "App de mensajería. Riesgo bajo si no genera ansiedad o dependencia."
    }
    msg = f"Tu plataforma más usada es {platform}. "
    return msg + risks.get(platform, "Plataforma no clasificada. Evalúa tu experiencia personal al usarla.")

def classify_social_media_usage(hours: float) -> str:
    """Clasifica el nivel de uso diario de redes sociales."""
    msg = f"Usas redes sociales {hours} horas al día. "
    if hours <= 2:
        return msg + "Uso bajo de redes sociales. Bien manejado."
    elif hours <= 4:
        return msg + "Uso moderado. Asegúrate de que no interfiera con tus actividades."
    elif hours <= 6:
        return msg + "Uso alto. Considera monitorear tu tiempo en redes."
    else:
        return msg + "Uso excesivo. Podría estar afectando tu bienestar o productividad."

def classify_conflicts(conflicts: int) -> str:
    """Evalúa los conflictos sociales en redes."""
    msg = f"Has reportado {conflicts} conflictos en redes sociales. "
    if conflicts == 0:
        return msg + "No reportas conflictos en redes sociales, lo cual es positivo."
    elif conflicts <= 2:
        return msg + "Has tenido algunos conflictos. Considera evitar discusiones innecesarias."
    else:
        return msg + "Has tenido múltiples conflictos. Esto puede afectar tu estado emocional."

def classify_addiction_score(score: float) -> str:
    """Clasifica el nivel de adicción en base al puntaje calculado."""
    msg = f"Tu puntaje de adicción es {score}. "
    if score <= 3:
        return msg + "Tu nivel de adicción es bajo. Tienes un buen control del tiempo en redes."
    elif score <= 6:
        return msg + "Tu nivel de adicción es moderado. Mantente alerta y cuida tu equilibrio digital."
    elif score <= 8:
        return msg + "Tu nivel de adicción es alto. Considera reducir el uso de redes."
    else:
        return msg + "Tu adicción a redes sociales es muy alta. Es importante hacer cambios en tu rutina."
    
def classify_academic_impact(affects: int) -> str:
    """Clasifica si el rendimiento académico podría estar afectado."""
    return "Tu estilo de vida digital podría estar afectando tus estudios." if affects else "No parece que tus hábitos digitales afecten tu rendimiento académico."

def get_personal_recommendations(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recibe los resultados de las funciones clasificadoras y genera recomendaciones
    personalizadas en base a los factores detectados.
    """

    risk_factors = []
    recommendations = []

    # Detectar factores de riesgo a partir de resultados previos
    if results["addicted_score"] > 6:
        risk_factors.append("Adicción elevada")

    if results["sleep_hours"] < 5:
        risk_factors.append("Pocas horas de sueño")

    if results["affects_academic"]:
        risk_factors.append("Afectación académica")

    if results["conflicts"] > 2:
        risk_factors.append("Conflictos frecuentes")

    if results["usage_hours"] > 6:
        risk_factors.append("Uso excesivo de redes sociales")

    if results["platform"] in ["TikTok", "Instagram"]:
        risk_factors.append(f"Uso de plataforma riesgosa ({results['platform']})")

    # Generar recomendaciones base según número de factores
    if len(risk_factors) >= 5:
        recommendations.extend([
            "Establece horarios fijos para desconectarte digitalmente.",
            "Busca acompañamiento profesional si sientes sobrecarga emocional.",
            "Evita discusiones innecesarias en redes y prioriza el descanso.",
        ])
    elif len(risk_factors) >= 3:
        recommendations.extend([
            "Reduce gradualmente el tiempo en redes sociales.",
            "Haz pausas activas y descansos visuales frecuentes.",
            "Revisa tus hábitos de sueño para mejorarlos."
        ])
    else:
        recommendations.extend([
            "Mantén tus hábitos actuales y sigue observando tu equilibrio digital.",
            "Continúa priorizando tu bienestar emocional y académico."
        ])

    # Recomendaciones específicas por factor
    if "Adicción elevada" in risk_factors:
        recommendations.append("Monitorea tu tiempo con apps como Digital Wellbeing o Forest.")

    if "Pocas horas de sueño" in risk_factors:
        recommendations.append("Crea una rutina de sueño estable y evita pantallas antes de dormir.")

    if "Afectación académica" in risk_factors:
        recommendations.append("Organiza tu día para tener tiempo claro para estudiar y descansar.")

    if "Conflictos frecuentes" in risk_factors:
        recommendations.append("Practica empatía y regula tus emociones en redes sociales.")

    if "Uso excesivo de redes sociales" in risk_factors:
        recommendations.append("Configura límites de uso diario para cada aplicación.")

    if any("riesgosa" in rf for rf in risk_factors):
        recommendations.append("Evalúa si la plataforma que más usas aporta valor a tu vida diaria.")

    return {
        "risk_factors": risk_factors,
        "recommendations": list(set(recommendations))  # Elimina duplicados
    }

def detect_outliers(data: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """Detectar outliers en una columna"""
    if column not in data.columns:
        return pd.Series(dtype=bool)
    
    if method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(data[column].dropna()))
        return z_scores > 3
    
    return pd.Series(dtype=bool)

def create_summary_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """Crear estadísticas resumen de un DataFrame"""
    try:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        summary = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'numeric_columns': len(numeric_columns),
            'categorical_columns': len(categorical_columns),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Resumen de columnas numéricas
        for col in numeric_columns:
            summary['numeric_summary'][col] = {
                'mean': float(data[col].mean()),
                'median': float(data[col].median()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max())
            }
        
        # Resumen de columnas categóricas
        for col in categorical_columns:
            value_counts = data[col].value_counts()
            summary['categorical_summary'][col] = {
                'unique_values': len(value_counts),
                'most_common': value_counts.head(5).to_dict(),
                'missing_count': int(data[col].isnull().sum())
            }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error creando estadísticas resumen: {str(e)}")
        return {}