import os
import pandas as pd
from datetime import datetime
import json
from typing import Dict, Any, List
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(log_file: str = "app.log"):
    """Configurar logging para la aplicación"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

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

def format_timestamp(timestamp: str = None) -> str:
    """Formatear timestamp para display"""
    if not timestamp:
        timestamp = datetime.now().isoformat()
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp

def calculate_correlation(data: pd.DataFrame, col1: str, col2: str) -> float:
    """Calcular correlación entre dos columnas"""
    try:
        if col1 in data.columns and col2 in data.columns:
            return float(data[col1].corr(data[col2]))
    except:
        pass
    return 0.0

def get_performance_category(score: float) -> str:
    """Categorizar rendimiento académico"""
    if score >= 90:
        return "Excelente"
    elif score >= 80:
        return "Bueno"
    elif score >= 70:
        return "Regular"
    elif score >= 60:
        return "Deficiente"
    else:
        return "Muy Deficiente"

def get_social_media_risk_level(usage_hours: int) -> str:
    """Determinar nivel de riesgo por uso de redes sociales"""
    if usage_hours <= 2:
        return "Bajo"
    elif usage_hours <= 4:
        return "Medio-Bajo"
    elif usage_hours <= 6:
        return "Medio"
    elif usage_hours <= 8:
        return "Medio-Alto"
    else:
        return "Alto"

def generate_user_insights(user_data: Dict) -> List[str]:
    """Generar insights sobre un usuario"""
    insights = []
    
    # Análisis de rendimiento
    performance = safe_float(user_data.get('academic_performance', 0))
    performance_cat = get_performance_category(performance)
    insights.append(f"Rendimiento académico: {performance_cat} ({performance:.1f}%)")
    
    # Análisis de uso de redes sociales
    social_usage = safe_int(user_data.get('social_media_usage', 0))
    risk_level = get_social_media_risk_level(social_usage)
    insights.append(f"Riesgo por redes sociales: {risk_level} ({social_usage} horas/día)")
    
    # Análisis de horas de estudio
    study_hours = safe_int(user_data.get('study_hours', 0))
    if study_hours < 15:
        insights.append("Horas de estudio por debajo del promedio recomendado")
    elif study_hours > 40:
        insights.append("Dedicación excepcional a los estudios")
    else:
        insights.append("Horas de estudio en rango normal")
    
    # Análisis de plataforma principal
    platform = user_data.get('main_platform', '')
    platform_insights = {
        'Instagram': 'Plataforma visual que puede ser distrayente',
        'TikTok': 'Contenido rápido que puede afectar la concentración',
        'YouTube': 'Plataforma con potencial educativo',
        'LinkedIn': 'Plataforma profesional, menor riesgo académico',
        'Facebook': 'Red social tradicional con riesgo medio',
        'Twitter': 'Información rápida, puede ser útil o distrayente'
    }
    
    if platform in platform_insights:
        insights.append(f"Plataforma principal: {platform} - {platform_insights[platform]}")
    
    return insights

def calculate_academic_risk_score(user_data: Dict) -> Dict[str, Any]:
    """Calcular puntaje de riesgo académico"""
    score = 0
    factors = []
    
    # Factor: Rendimiento académico actual
    performance = safe_float(user_data.get('academic_performance', 75))
    if performance < 60:
        score += 30
        factors.append("Rendimiento académico bajo")
    elif performance < 75:
        score += 15
        factors.append("Rendimiento académico regular")
    
    # Factor: Uso excesivo de redes sociales
    social_usage = safe_int(user_data.get('social_media_usage', 5))
    if social_usage > 7:
        score += 25
        factors.append("Uso excesivo de redes sociales")
    elif social_usage > 5:
        score += 10
        factors.append("Uso moderado-alto de redes sociales")
    
    # Factor: Pocas horas de estudio
    study_hours = safe_int(user_data.get('study_hours', 25))
    if study_hours < 15:
        score += 20
        factors.append("Pocas horas de estudio")
    elif study_hours < 25:
        score += 10
        factors.append("Horas de estudio por debajo del promedio")
    
    # Factor: Plataforma de alto riesgo
    platform = user_data.get('main_platform', '')
    high_risk_platforms = ['TikTok', 'Instagram']
    if platform in high_risk_platforms:
        score += 15
        factors.append(f"Uso de plataforma de alto riesgo ({platform})")
    
    # Determinar nivel de riesgo
    if score >= 50:
        risk_level = "Muy Alto"
    elif score >= 35:
        risk_level = "Alto"
    elif score >= 20:
        risk_level = "Medio"
    elif score >= 10:
        risk_level = "Bajo"
    else:
        risk_level = "Muy Bajo"
    
    return {
        'risk_score': score,
        'risk_level': risk_level,
        'risk_factors': factors,
        'recommendations': generate_risk_recommendations(score, factors)
    }

def generate_risk_recommendations(risk_score: int, risk_factors: List[str]) -> List[str]:
    """Generar recomendaciones basadas en factores de riesgo"""
    recommendations = []
    
    if risk_score >= 35:
        recommendations.extend([
            "Buscar apoyo académico inmediato",
            "Reducir drásticamente el tiempo en redes sociales",
            "Implementar técnicas de estudio más efectivas",
            "Considerar asesoramiento psicopedagógico"
        ])
    elif risk_score >= 20:
        recommendations.extend([
            "Monitorear el tiempo en redes sociales",
            "Incrementar las horas de estudio",
            "Usar técnicas de gestión del tiempo",
            "Establecer metas académicas claras"
        ])
    else:
        recommendations.extend([
            "Mantener el equilibrio actual",
            "Optimizar métodos de estudio",
            "Continuar con hábitos saludables"
        ])
    
    # Recomendaciones específicas por factor
    if "Uso excesivo de redes sociales" in risk_factors:
        recommendations.append("Usar aplicaciones de control de tiempo de pantalla")
    
    if "Pocas horas de estudio" in risk_factors:
        recommendations.append("Crear un horario de estudio estructurado")
    
    if "Rendimiento académico bajo" in risk_factors:
        recommendations.append("Revisar métodos de estudio y buscar ayuda adicional")
    
    return list(set(recommendations))  # Eliminar duplicados

def export_data_to_json(data: Any, filename: str = None) -> str:
    """Exportar datos a JSON"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"export_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return filename
    except Exception as e:
        logger.error(f"Error exportando datos: {str(e)}")
        return ""

def validate_file_permissions(file_path: str) -> bool:
    """Validar permisos de archivo"""
    try:
        # Verificar si el directorio existe y es escribible
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Verificar permisos de escritura
        return os.access(directory, os.W_OK)
    except Exception as e:
        logger.error(f"Error validando permisos: {str(e)}")
        return False

def clean_numeric_data(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Limpiar datos numéricos en DataFrame"""
    cleaned_data = data.copy()
    
    for col in columns:
        if col in cleaned_data.columns:
            # Convertir a numérico, forzando errores a NaN
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            
            # Rellenar NaN con la mediana
            if cleaned_data[col].notna().any():
                median_value = cleaned_data[col].median()
                cleaned_data[col].fillna(median_value, inplace=True)
    
    return cleaned_data

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

def format_error_response(error: Exception, context: str = "") -> Dict[str, str]:
    """Formatear respuesta de error para API"""
    error_message = str(error)
    
    # Log del error
    logger.error(f"Error en {context}: {error_message}")
    
    return {
        'error': 'Error interno del servidor',
        'message': error_message,
        'context': context,
        'timestamp': datetime.now().isoformat()
    }

def get_file_size_mb(file_path: str) -> float:
    """Obtener tamaño de archivo en MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0
