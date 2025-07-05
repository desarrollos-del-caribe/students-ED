from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Union

@dataclass
class User:
    """Modelo de datos para Usuario"""
    id: Optional[int] = None
    name: str = ""
    email: str = ""
    age: int = 0
    gender: str = ""  # Masculino, Femenino, Otro
    education_level: str = ""  # Bachillerato, Universidad, Posgrado
    social_media_usage: int = 0  # 1-10 horas diarias
    academic_performance: float = 0.0  # 0-100
    main_platform: str = ""  # Instagram, TikTok, Facebook, Twitter, YouTube, LinkedIn
    study_hours: int = 0  # horas semanales
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'age': self.age,
            'gender': self.gender,
            'education_level': self.education_level,
            'social_media_usage': self.social_media_usage,
            'academic_performance': self.academic_performance,
            'main_platform': self.main_platform,
            'study_hours': self.study_hours,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'User':
        """Crear usuario desde diccionario"""
        return cls(
            id=data.get('id'),
            name=data.get('name', ''),
            email=data.get('email', ''),
            age=data.get('age', 0),
            gender=data.get('gender', ''),
            education_level=data.get('education_level', ''),
            social_media_usage=data.get('social_media_usage', 0),
            academic_performance=data.get('academic_performance', 0.0),
            main_platform=data.get('main_platform', ''),
            study_hours=data.get('study_hours', 0),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
    
    def validate(self) -> List[str]:
        """Validar datos del usuario"""
        errors = []
        
        if not self.name:
            errors.append("Nombre es requerido")
        
        if not self.email:
            errors.append("Email es requerido")
        
        if self.age < 16 or self.age > 100:
            errors.append("Edad debe estar entre 16 y 100 años")
        
        if self.gender not in ['Masculino', 'Femenino', 'Otro']:
            errors.append("Género debe ser: Masculino, Femenino, u Otro")
        
        if self.education_level not in ['Bachillerato', 'Universidad', 'Posgrado']:
            errors.append("Nivel educativo debe ser: Bachillerato, Universidad, o Posgrado")
        
        if self.social_media_usage < 1 or self.social_media_usage > 10:
            errors.append("Uso de redes sociales debe ser entre 1 y 10 horas")
        
        if self.academic_performance < 0 or self.academic_performance > 100:
            errors.append("Rendimiento académico debe ser entre 0 y 100")
        
        valid_platforms = ['Instagram', 'TikTok', 'Facebook', 'Twitter', 'YouTube', 'LinkedIn']
        if self.main_platform not in valid_platforms:
            errors.append(f"Plataforma principal debe ser una de: {', '.join(valid_platforms)}")
        
        if self.study_hours < 0 or self.study_hours > 168:  # 168 horas en una semana
            errors.append("Horas de estudio debe ser entre 0 y 168 horas semanales")
        
        return errors


@dataclass
class MLModel:
    """Modelo de datos para Modelo ML"""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    category: str = ""  # supervised, unsupervised, ensemble
    difficulty: str = ""  # Principiante, Intermedio, Avanzado
    is_locked: bool = True
    unlock_condition: str = ""
    estimated_time: int = 0  # minutos
    use_cases: str = ""  # separado por comas
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'difficulty': self.difficulty,
            'is_locked': self.is_locked,
            'unlock_condition': self.unlock_condition,
            'estimated_time': self.estimated_time,
            'use_cases': self.use_cases
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MLModel':
        """Crear modelo desde diccionario"""
        return cls(
            id=data.get('id'),
            name=data.get('name', ''),
            description=data.get('description', ''),
            category=data.get('category', ''),
            difficulty=data.get('difficulty', ''),
            is_locked=data.get('is_locked', True),
            unlock_condition=data.get('unlock_condition', ''),
            estimated_time=data.get('estimated_time', 0),
            use_cases=data.get('use_cases', '')
        )
    
    def get_use_cases_list(self) -> List[str]:
        """Obtener lista de casos de uso"""
        return [case.strip() for case in self.use_cases.split(',') if case.strip()]


@dataclass
class Prediction:
    """Modelo de datos para Predicción"""
    id: Optional[int] = None
    user_id: int = 0
    model_id: int = 0
    prediction_result: Union[str, Dict] = ""
    accuracy: Optional[float] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'model_id': self.model_id,
            'prediction_result': self.prediction_result,
            'accuracy': self.accuracy,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Prediction':
        """Crear predicción desde diccionario"""
        return cls(
            id=data.get('id'),
            user_id=data.get('user_id', 0),
            model_id=data.get('model_id', 0),
            prediction_result=data.get('prediction_result', ''),
            accuracy=data.get('accuracy'),
            created_at=data.get('created_at')
        )


@dataclass
class MLResults:
    """Modelo de datos para resultados de ML"""
    status: str = "success"  # success, error
    model_id: str = ""
    user_id: str = ""
    results: Dict = None
    metrics: Dict = None
    visualizations: Dict = None
    interpretation: Dict = None
    timestamp: str = ""
    
    def __post_init__(self):
        if self.results is None:
            self.results = {}
        if self.metrics is None:
            self.metrics = {}
        if self.visualizations is None:
            self.visualizations = {}
        if self.interpretation is None:
            self.interpretation = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'status': self.status,
            'model_id': self.model_id,
            'user_id': self.user_id,
            'results': self.results,
            'metrics': self.metrics,
            'visualizations': self.visualizations,
            'interpretation': self.interpretation,
            'timestamp': self.timestamp
        }


@dataclass
class UserProfile:
    """Modelo de datos para perfil de usuario analizado"""
    user_id: int = 0
    risk_level: str = ""  # Bajo, Medio, Alto
    academic_prediction: float = 0.0
    usage_pattern: str = ""
    dominant_factors: List[str] = None
    
    def __post_init__(self):
        if self.dominant_factors is None:
            self.dominant_factors = []
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'user_id': self.user_id,
            'risk_level': self.risk_level,
            'academic_prediction': self.academic_prediction,
            'usage_pattern': self.usage_pattern,
            'dominant_factors': self.dominant_factors
        }


@dataclass
class ModelComparison:
    """Modelo de datos para comparación de modelos"""
    model_name: str = ""
    prediction: Union[float, str] = ""
    confidence: float = 0.0
    key_factors: List[str] = None
    
    def __post_init__(self):
        if self.key_factors is None:
            self.key_factors = []
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'model_name': self.model_name,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'key_factors': self.key_factors
        }


@dataclass
class Recommendations:
    """Modelo de datos para recomendaciones"""
    immediate_actions: List[str] = None
    long_term_strategies: List[str] = None
    resource_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.immediate_actions is None:
            self.immediate_actions = []
        if self.long_term_strategies is None:
            self.long_term_strategies = []
        if self.resource_suggestions is None:
            self.resource_suggestions = []
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'immediate_actions': self.immediate_actions,
            'long_term_strategies': self.long_term_strategies,
            'resource_suggestions': self.resource_suggestions
        }


@dataclass
class AnalysisResults:
    """Modelo de datos para resultados de análisis completo"""
    user_profile: UserProfile = None
    model_comparisons: List[ModelComparison] = None
    recommendations: Recommendations = None
    
    def __post_init__(self):
        if self.user_profile is None:
            self.user_profile = UserProfile()
        if self.model_comparisons is None:
            self.model_comparisons = []
        if self.recommendations is None:
            self.recommendations = Recommendations()
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'user_profile': self.user_profile.to_dict(),
            'model_comparisons': [comp.to_dict() for comp in self.model_comparisons],
            'recommendations': self.recommendations.to_dict()
        }
