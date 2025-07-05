import pandas as pd
import openpyxl
from openpyxl import Workbook
import os
from datetime import datetime
import shutil
from typing import Dict, List, Optional, Union
import numpy as np

class ExcelHandler:
    """Clase principal para manejar todas las operaciones con archivos Excel"""
    
    def __init__(self, base_path: str = "./data/excel"):
        self.base_path = base_path
        self.backup_path = os.path.join(base_path, "backups")
        self.ensure_directories_exist()
        
    def ensure_directories_exist(self):
        """Crear directorios si no existen"""
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
    
    def get_file_path(self, filename: str) -> str:
        """Obtener la ruta completa del archivo"""
        return os.path.join(self.base_path, filename)
    
    def backup_excel_file(self, filename: str) -> str:
        """Crear backup de archivo Excel con timestamp"""
        source_path = self.get_file_path(filename)
        if os.path.exists(source_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{filename.split('.')[0]}_{timestamp}.xlsx"
            backup_path = os.path.join(self.backup_path, backup_filename)
            shutil.copy2(source_path, backup_path)
            return backup_path
        return None
    
    def file_exists(self, filename: str) -> bool:
        """Verificar si el archivo existe"""
        return os.path.exists(self.get_file_path(filename))
    
    def read_excel_data(self, filename: str, sheet_name: str = None) -> pd.DataFrame:
        """Leer datos de archivo Excel"""
        file_path = self.get_file_path(filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo {filename} no encontrado")
        
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            return df
        except Exception as e:
            raise Exception(f"Error al leer archivo {filename}: {str(e)}")
    
    def write_excel_data(self, filename: str, data: pd.DataFrame, sheet_name: str = "Sheet1", backup: bool = True):
        """Escribir datos a archivo Excel"""
        file_path = self.get_file_path(filename)
        
        # Crear backup si existe y se solicita
        if backup and os.path.exists(file_path):
            self.backup_excel_file(filename)
        
        try:
            data.to_excel(file_path, sheet_name=sheet_name, index=False)
        except Exception as e:
            raise Exception(f"Error al escribir archivo {filename}: {str(e)}")
    
    def append_excel_row(self, filename: str, new_data: Dict, backup: bool = True):
        """Agregar nueva fila a archivo Excel existente"""
        try:
            df = self.read_excel_data(filename)
        except FileNotFoundError:
            # Si el archivo no existe, crear uno nuevo
            df = pd.DataFrame()
        
        # Crear backup si existe
        if backup and self.file_exists(filename):
            self.backup_excel_file(filename)
        
        # Agregar nueva fila
        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        self.write_excel_data(filename, df, backup=False)
        return df
    
    def update_excel_row(self, filename: str, condition: Dict, new_data: Dict, backup: bool = True):
        """Actualizar fila específica en archivo Excel"""
        df = self.read_excel_data(filename)
        
        # Crear backup
        if backup:
            self.backup_excel_file(filename)
        
        # Encontrar y actualizar fila
        mask = pd.Series([True] * len(df))
        for column, value in condition.items():
            mask &= (df[column] == value)
        
        if mask.any():
            for column, value in new_data.items():
                df.loc[mask, column] = value
            
            self.write_excel_data(filename, df, backup=False)
            return df
        else:
            raise ValueError("No se encontró ninguna fila que coincida con la condición")
    
    def validate_excel_structure(self, filename: str, required_columns: List[str]) -> bool:
        """Validar que el archivo Excel tenga las columnas requeridas"""
        try:
            df = self.read_excel_data(filename)
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
            return True
        except Exception as e:
            raise Exception(f"Error validando estructura de {filename}: {str(e)}")
    
    def create_excel_with_headers(self, filename: str, headers: List[str]):
        """Crear archivo Excel con headers específicos"""
        df = pd.DataFrame(columns=headers)
        self.write_excel_data(filename, df, backup=False)
        return df


class UsersExcelHandler(ExcelHandler):
    """Manejador específico para el archivo users.xlsx"""
    
    FILENAME = "users.xlsx"
    REQUIRED_COLUMNS = [
        'id', 'name', 'email', 'age', 'gender', 'education_level',
        'social_media_usage', 'academic_performance', 'main_platform',
        'study_hours', 'created_at', 'updated_at'
    ]
    
    def __init__(self, base_path: str = "./data/excel"):
        super().__init__(base_path)
        self.initialize_if_not_exists()
    
    def initialize_if_not_exists(self):
        """Inicializar archivo si no existe"""
        if not self.file_exists(self.FILENAME):
            self.create_excel_with_headers(self.FILENAME, self.REQUIRED_COLUMNS)
            self.populate_sample_data()
    
    def populate_sample_data(self):
        """Poblar con datos de ejemplo"""
        sample_users = [
            {
                'id': 1,
                'name': 'Ana García',
                'email': 'ana.garcia@email.com',
                'age': 20,
                'gender': 'Femenino',
                'education_level': 'Universidad',
                'social_media_usage': 6,
                'academic_performance': 78.5,
                'main_platform': 'Instagram',
                'study_hours': 25,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            },
            {
                'id': 2,
                'name': 'Carlos López',
                'email': 'carlos.lopez@email.com',
                'age': 22,
                'gender': 'Masculino',
                'education_level': 'Universidad',
                'social_media_usage': 3,
                'academic_performance': 85.2,
                'main_platform': 'LinkedIn',
                'study_hours': 35,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            },
            {
                'id': 3,
                'name': 'María Silva',
                'email': 'maria.silva@email.com',
                'age': 19,
                'gender': 'Femenino',
                'education_level': 'Universidad',
                'social_media_usage': 8,
                'academic_performance': 72.0,
                'main_platform': 'TikTok',
                'study_hours': 20,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            },
            {
                'id': 4,
                'name': 'Juan Pérez',
                'email': 'juan.perez@email.com',
                'age': 21,
                'gender': 'Masculino',
                'education_level': 'Universidad',
                'social_media_usage': 4,
                'academic_performance': 82.5,
                'main_platform': 'YouTube',
                'study_hours': 30,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
        ]
        
        df = pd.DataFrame(sample_users)
        self.write_excel_data(self.FILENAME, df, backup=False)
    
    def read_users(self) -> pd.DataFrame:
        """Leer todos los usuarios"""
        return self.read_excel_data(self.FILENAME)
    
    def get_user_by_id(self, user_id: int) -> Dict:
        """Obtener usuario específico por ID"""
        df = self.read_users()
        user_row = df[df['id'] == user_id]
        if user_row.empty:
            raise ValueError(f"Usuario con ID {user_id} no encontrado")
        return user_row.iloc[0].to_dict()
    
    def create_user(self, user_data: Dict) -> Dict:
        """Crear nuevo usuario"""
        df = self.read_users()
        
        # Generar nuevo ID
        if len(df) > 0:
            new_id = df['id'].max() + 1
        else:
            new_id = 1
        
        # Preparar datos del usuario
        user_data['id'] = new_id
        user_data['created_at'] = datetime.now().isoformat()
        user_data['updated_at'] = datetime.now().isoformat()
        
        # Validar datos requeridos
        self.validate_user_data(user_data)
        
        self.append_excel_row(self.FILENAME, user_data)
        return user_data
    
    def update_user(self, user_id: int, user_data: Dict) -> Dict:
        """Actualizar usuario existente"""
        user_data['updated_at'] = datetime.now().isoformat()
        self.update_excel_row(self.FILENAME, {'id': user_id}, user_data)
        return self.get_user_by_id(user_id)
    
    def validate_user_data(self, user_data: Dict):
        """Validar datos del usuario"""
        required_fields = ['name', 'email', 'age', 'gender', 'education_level']
        for field in required_fields:
            if field not in user_data:
                raise ValueError(f"Campo requerido faltante: {field}")
        
        # Validar valores específicos
        if user_data.get('gender') not in ['Masculino', 'Femenino', 'Otro']:
            raise ValueError("Género debe ser: Masculino, Femenino, u Otro")
        
        if user_data.get('education_level') not in ['Bachillerato', 'Universidad', 'Posgrado']:
            raise ValueError("Nivel educativo debe ser: Bachillerato, Universidad, o Posgrado")
        
        if user_data.get('social_media_usage') and not (1 <= user_data['social_media_usage'] <= 10):
            raise ValueError("Uso de redes sociales debe ser entre 1 y 10 horas")
        
        if user_data.get('academic_performance') and not (0 <= user_data['academic_performance'] <= 100):
            raise ValueError("Rendimiento académico debe ser entre 0 y 100")


class ModelsExcelHandler(ExcelHandler):
    """Manejador específico para el archivo ml_models.xlsx"""
    
    FILENAME = "ml_models.xlsx"
    REQUIRED_COLUMNS = [
        'id', 'name', 'description', 'category', 'difficulty',
        'is_locked', 'unlock_condition', 'estimated_time', 'use_cases'
    ]
    
    def __init__(self, base_path: str = "./data/excel"):
        super().__init__(base_path)
        self.initialize_if_not_exists()
    
    def initialize_if_not_exists(self):
        """Inicializar archivo si no existe"""
        if not self.file_exists(self.FILENAME):
            self.create_excel_with_headers(self.FILENAME, self.REQUIRED_COLUMNS)
            self.populate_sample_models()
    
    def populate_sample_models(self):
        """Poblar con modelos de ejemplo"""
        sample_models = [
            {
                'id': 1,
                'name': 'Regresión Lineal',
                'description': 'Predice el rendimiento académico basado en variables continuas',
                'category': 'supervised',
                'difficulty': 'Principiante',
                'is_locked': False,
                'unlock_condition': 'Ninguna',
                'estimated_time': 5,
                'use_cases': 'predicción continua,análisis de tendencias'
            },
            {
                'id': 2,
                'name': 'Regresión Logística',
                'description': 'Clasifica el riesgo de bajo rendimiento académico',
                'category': 'supervised',
                'difficulty': 'Principiante',
                'is_locked': False,
                'unlock_condition': 'Ninguna',
                'estimated_time': 5,
                'use_cases': 'clasificación binaria,análisis de riesgo'
            },
            {
                'id': 3,
                'name': 'K-Means Clustering',
                'description': 'Agrupa estudiantes por patrones de comportamiento',
                'category': 'unsupervised',
                'difficulty': 'Intermedio',
                'is_locked': True,
                'unlock_condition': 'Completar formulario de usuario',
                'estimated_time': 10,
                'use_cases': 'segmentación,patrones de comportamiento'
            },
            {
                'id': 4,
                'name': 'Random Forest',
                'description': 'Predicción compleja usando múltiples árboles de decisión',
                'category': 'ensemble',
                'difficulty': 'Avanzado',
                'is_locked': True,
                'unlock_condition': 'Completar formulario de usuario',
                'estimated_time': 15,
                'use_cases': 'predicción compleja,importancia de características'
            },
            {
                'id': 5,
                'name': 'Árboles de Decisión',
                'description': 'Genera reglas de decisión interpretables',
                'category': 'supervised',
                'difficulty': 'Intermedio',
                'is_locked': True,
                'unlock_condition': 'Completar formulario de usuario',
                'estimated_time': 8,
                'use_cases': 'reglas de negocio,decisiones interpretables'
            },
            {
                'id': 6,
                'name': 'Support Vector Machine',
                'description': 'Clasificación avanzada con márgenes de decisión',
                'category': 'supervised',
                'difficulty': 'Avanzado',
                'is_locked': True,
                'unlock_condition': 'Completar formulario de usuario',
                'estimated_time': 12,
                'use_cases': 'clasificación compleja,márgenes de decisión'
            }
        ]
        
        df = pd.DataFrame(sample_models)
        self.write_excel_data(self.FILENAME, df, backup=False)
    
    def read_models(self) -> pd.DataFrame:
        """Leer todos los modelos"""
        return self.read_excel_data(self.FILENAME)
    
    def get_model_by_id(self, model_id: int) -> Dict:
        """Obtener modelo específico por ID"""
        df = self.read_models()
        model_row = df[df['id'] == model_id]
        if model_row.empty:
            raise ValueError(f"Modelo con ID {model_id} no encontrado")
        return model_row.iloc[0].to_dict()


class PredictionsExcelHandler(ExcelHandler):
    """Manejador específico para el archivo predictions.xlsx"""
    
    FILENAME = "predictions.xlsx"
    REQUIRED_COLUMNS = [
        'id', 'user_id', 'model_id', 'prediction_result', 'accuracy', 'created_at'
    ]
    
    def __init__(self, base_path: str = "./data/excel"):
        super().__init__(base_path)
        self.initialize_if_not_exists()
    
    def initialize_if_not_exists(self):
        """Inicializar archivo si no existe"""
        if not self.file_exists(self.FILENAME):
            self.create_excel_with_headers(self.FILENAME, self.REQUIRED_COLUMNS)
    
    def save_prediction(self, user_id: int, model_id: int, prediction_result: Union[str, Dict], accuracy: float = None) -> Dict:
        """Guardar resultado de predicción"""
        df = self.read_excel_data(self.FILENAME) if self.file_exists(self.FILENAME) else pd.DataFrame()
        
        # Generar nuevo ID
        if len(df) > 0:
            new_id = df['id'].max() + 1
        else:
            new_id = 1
        
        prediction_data = {
            'id': new_id,
            'user_id': user_id,
            'model_id': model_id,
            'prediction_result': str(prediction_result) if isinstance(prediction_result, dict) else prediction_result,
            'accuracy': accuracy,
            'created_at': datetime.now().isoformat()
        }
        
        self.append_excel_row(self.FILENAME, prediction_data)
        return prediction_data
    
    def get_user_predictions(self, user_id: int) -> List[Dict]:
        """Obtener todas las predicciones de un usuario"""
        try:
            df = self.read_excel_data(self.FILENAME)
            user_predictions = df[df['user_id'] == user_id]
            return user_predictions.to_dict('records')
        except FileNotFoundError:
            return []
    
    def get_model_results(self, model_id: int, user_id: int = None) -> List[Dict]:
        """Obtener resultados de un modelo específico"""
        try:
            df = self.read_excel_data(self.FILENAME)
            results = df[df['model_id'] == model_id]
            if user_id:
                results = results[results['user_id'] == user_id]
            return results.to_dict('records')
        except FileNotFoundError:
            return []
