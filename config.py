import os
from datetime import datetime

class Config:
    """Configuración principal de la aplicación"""
    
    # Configuración básica
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Configuración de archivos Excel
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXCEL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'excel')
    EXCEL_BACKUP_DIR = os.path.join(EXCEL_DATA_DIR, 'backups')
    
    # Archivos Excel
    USERS_EXCEL_FILE = 'users.xlsx'
    MODELS_EXCEL_FILE = 'ml_models.xlsx'
    PREDICTIONS_EXCEL_FILE = 'predictions.xlsx'
    
    # Configuración de CORS
    CORS_ORIGINS = [
        'http://localhost:3000',
        'http://localhost:5173',
        'http://localhost:8080',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:5173'
    ]
    
    # Configuración de Rate Limiting
    RATELIMIT_STORAGE_URL = 'memory://'
    RATELIMIT_DEFAULT = "1000 per day, 200 per hour"
    
    # Configuración de ML
    ML_MODEL_CONFIGS = {
        1: {
            'name': 'Regresión Lineal',
            'type': 'regression',
            'target': 'academic_performance'
        },
        2: {
            'name': 'Regresión Logística',
            'type': 'classification',
            'target': 'risk_category'
        },
        3: {
            'name': 'K-Means Clustering',
            'type': 'clustering',
            'target': None
        },
        4: {
            'name': 'Random Forest',
            'type': 'regression',
            'target': 'academic_performance'
        },
        5: {
            'name': 'Árboles de Decisión',
            'type': 'regression',
            'target': 'academic_performance'
        },
        6: {
            'name': 'Support Vector Machine',
            'type': 'classification',
            'target': 'performance_category'
        }
    }
    
    # Validaciones
    VALID_GENDERS = ['Masculino', 'Femenino', 'Otro']
    VALID_EDUCATION_LEVELS = ['Bachillerato', 'Universidad', 'Posgrado']
    VALID_PLATFORMS = ['Instagram', 'TikTok', 'Facebook', 'Twitter', 'YouTube', 'LinkedIn']
    VALID_MODEL_CATEGORIES = ['supervised', 'unsupervised', 'ensemble']
    VALID_DIFFICULTIES = ['Principiante', 'Intermedio', 'Avanzado']
    
    # Rangos de valores
    MIN_AGE = 16
    MAX_AGE = 100
    MIN_SOCIAL_MEDIA_USAGE = 1
    MAX_SOCIAL_MEDIA_USAGE = 10
    MIN_ACADEMIC_PERFORMANCE = 0
    MAX_ACADEMIC_PERFORMANCE = 100
    MIN_STUDY_HOURS = 0
    MAX_STUDY_HOURS = 168  # Horas en una semana
    
    # Configuración de backup
    BACKUP_RETENTION_DAYS = 30
    AUTO_BACKUP_ENABLED = True
    
    @staticmethod
    def init_app(app):
        """Inicializar configuración de la aplicación"""
        # Crear directorios si no existen
        os.makedirs(Config.EXCEL_DATA_DIR, exist_ok=True)
        os.makedirs(Config.EXCEL_BACKUP_DIR, exist_ok=True)
        
        # Configurar logging
        import logging
        logging.basicConfig(
            level=logging.INFO if not Config.DEBUG else logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @staticmethod
    def get_excel_file_path(filename: str) -> str:
        """Obtener ruta completa de archivo Excel"""
        return os.path.join(Config.EXCEL_DATA_DIR, filename)
    
    @staticmethod
    def get_backup_file_path(filename: str) -> str:
        """Obtener ruta de backup con timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        backup_filename = f"{name}_{timestamp}{ext}"
        return os.path.join(Config.EXCEL_BACKUP_DIR, backup_filename)


class DevelopmentConfig(Config):
    """Configuración para desarrollo"""
    DEBUG = True
    RATELIMIT_ENABLED = False


class ProductionConfig(Config):
    """Configuración para producción"""
    DEBUG = False
    RATELIMIT_ENABLED = True
    
    # En producción, usar variables de entorno
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY debe estar definida en producción")


class TestingConfig(Config):
    """Configuración para testing"""
    TESTING = True
    DEBUG = True
    RATELIMIT_ENABLED = False
    
    # Usar directorios de test
    EXCEL_DATA_DIR = os.path.join(Config.BASE_DIR, 'tests', 'data')
    EXCEL_BACKUP_DIR = os.path.join(EXCEL_DATA_DIR, 'backups')


# Configuración por defecto
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
