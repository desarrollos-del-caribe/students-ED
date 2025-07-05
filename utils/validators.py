import re
from typing import Dict, List, Any, Union
import pandas as pd
from datetime import datetime

class DataValidator:
    """Clase para validar datos de entrada"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validar formato de email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_age(age: int) -> bool:
        """Validar edad"""
        return 16 <= age <= 100
    
    @staticmethod
    def validate_gender(gender: str) -> bool:
        """Validar género"""
        valid_genders = ['Masculino', 'Femenino', 'Otro']
        return gender in valid_genders
    
    @staticmethod
    def validate_education_level(education_level: str) -> bool:
        """Validar nivel educativo"""
        valid_levels = ['Bachillerato', 'Universidad', 'Posgrado']
        return education_level in valid_levels
    
    @staticmethod
    def validate_social_media_usage(usage: int) -> bool:
        """Validar uso de redes sociales"""
        return 1 <= usage <= 10
    
    @staticmethod
    def validate_academic_performance(performance: float) -> bool:
        """Validar rendimiento académico"""
        return 0 <= performance <= 100
    
    @staticmethod
    def validate_main_platform(platform: str) -> bool:
        """Validar plataforma principal"""
        valid_platforms = ['Instagram', 'TikTok', 'Facebook', 'Twitter', 'YouTube', 'LinkedIn']
        return platform in valid_platforms
    
    @staticmethod
    def validate_study_hours(hours: int) -> bool:
        """Validar horas de estudio"""
        return 0 <= hours <= 168  # 168 horas en una semana
    
    @staticmethod
    def validate_user_data(data: Dict) -> Dict[str, List[str]]:
        """Validar datos completos de usuario"""
        errors = {
            'required_fields': [],
            'format_errors': [],
            'value_errors': []
        }
        
        # Campos requeridos
        required_fields = ['name', 'email', 'age', 'gender', 'education_level']
        for field in required_fields:
            if field not in data or not data[field]:
                errors['required_fields'].append(f'Campo requerido: {field}')
        
        # Validaciones de formato
        if 'email' in data and data['email']:
            if not DataValidator.validate_email(data['email']):
                errors['format_errors'].append('Formato de email inválido')
        
        if 'name' in data and data['name']:
            if len(data['name'].strip()) < 2:
                errors['format_errors'].append('Nombre debe tener al menos 2 caracteres')
        
        # Validaciones de valores
        if 'age' in data:
            try:
                age = int(data['age'])
                if not DataValidator.validate_age(age):
                    errors['value_errors'].append('Edad debe estar entre 16 y 100 años')
            except (ValueError, TypeError):
                errors['format_errors'].append('Edad debe ser un número entero')
        
        if 'gender' in data and data['gender']:
            if not DataValidator.validate_gender(data['gender']):
                errors['value_errors'].append('Género debe ser: Masculino, Femenino, u Otro')
        
        if 'education_level' in data and data['education_level']:
            if not DataValidator.validate_education_level(data['education_level']):
                errors['value_errors'].append('Nivel educativo debe ser: Bachillerato, Universidad, o Posgrado')
        
        if 'social_media_usage' in data:
            try:
                usage = int(data['social_media_usage'])
                if not DataValidator.validate_social_media_usage(usage):
                    errors['value_errors'].append('Uso de redes sociales debe ser entre 1 y 10 horas')
            except (ValueError, TypeError):
                errors['format_errors'].append('Uso de redes sociales debe ser un número entero')
        
        if 'academic_performance' in data:
            try:
                performance = float(data['academic_performance'])
                if not DataValidator.validate_academic_performance(performance):
                    errors['value_errors'].append('Rendimiento académico debe ser entre 0 y 100')
            except (ValueError, TypeError):
                errors['format_errors'].append('Rendimiento académico debe ser un número')
        
        if 'main_platform' in data and data['main_platform']:
            if not DataValidator.validate_main_platform(data['main_platform']):
                errors['value_errors'].append('Plataforma debe ser: Instagram, TikTok, Facebook, Twitter, YouTube, o LinkedIn')
        
        if 'study_hours' in data:
            try:
                hours = int(data['study_hours'])
                if not DataValidator.validate_study_hours(hours):
                    errors['value_errors'].append('Horas de estudio debe ser entre 0 y 168 horas semanales')
            except (ValueError, TypeError):
                errors['format_errors'].append('Horas de estudio debe ser un número entero')
        
        return errors
    
    @staticmethod
    def has_errors(validation_result: Dict[str, List[str]]) -> bool:
        """Verificar si hay errores en la validación"""
        return any(errors for errors in validation_result.values())
    
    @staticmethod
    def get_all_errors(validation_result: Dict[str, List[str]]) -> List[str]:
        """Obtener todos los errores como lista plana"""
        all_errors = []
        for error_type, errors in validation_result.items():
            all_errors.extend(errors)
        return all_errors


class ExcelValidator:
    """Clase para validar estructura de archivos Excel"""
    
    @staticmethod
    def validate_excel_structure(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
        """Validar estructura de DataFrame"""
        validation_result = {
            'is_valid': True,
            'missing_columns': [],
            'extra_columns': [],
            'data_issues': [],
            'summary': {}
        }
        
        # Verificar columnas faltantes
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_result['missing_columns'] = list(missing_columns)
            validation_result['is_valid'] = False
        
        # Verificar columnas extra
        extra_columns = set(df.columns) - set(required_columns)
        if extra_columns:
            validation_result['extra_columns'] = list(extra_columns)
        
        # Verificar datos faltantes
        missing_data = df.isnull().sum()
        if missing_data.any():
            validation_result['data_issues'].append(f'Datos faltantes: {missing_data.to_dict()}')
        
        # Verificar filas duplicadas
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_result['data_issues'].append(f'Filas duplicadas: {duplicates}')
        
        # Resumen
        validation_result['summary'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': duplicates
        }
        
        return validation_result
    
    @staticmethod
    def validate_users_excel(df: pd.DataFrame) -> Dict[str, Any]:
        """Validar específicamente archivo de usuarios"""
        required_columns = [
            'id', 'name', 'email', 'age', 'gender', 'education_level',
            'social_media_usage', 'academic_performance', 'main_platform',
            'study_hours', 'created_at', 'updated_at'
        ]
        
        validation = ExcelValidator.validate_excel_structure(df, required_columns)
        
        if not validation['missing_columns']:
            # Validaciones específicas de usuarios
            data_issues = validation['data_issues']
            
            # Validar tipos de datos
            if 'age' in df.columns:
                invalid_ages = df[(df['age'] < 16) | (df['age'] > 100)]
                if len(invalid_ages) > 0:
                    data_issues.append(f'Edades inválidas: {len(invalid_ages)} registros')
            
            if 'social_media_usage' in df.columns:
                invalid_usage = df[(df['social_media_usage'] < 1) | (df['social_media_usage'] > 10)]
                if len(invalid_usage) > 0:
                    data_issues.append(f'Uso de redes sociales inválido: {len(invalid_usage)} registros')
            
            if 'academic_performance' in df.columns:
                invalid_performance = df[(df['academic_performance'] < 0) | (df['academic_performance'] > 100)]
                if len(invalid_performance) > 0:
                    data_issues.append(f'Rendimiento académico inválido: {len(invalid_performance)} registros')
            
            if 'gender' in df.columns:
                valid_genders = ['Masculino', 'Femenino', 'Otro']
                invalid_genders = df[~df['gender'].isin(valid_genders)]
                if len(invalid_genders) > 0:
                    data_issues.append(f'Géneros inválidos: {len(invalid_genders)} registros')
            
            if 'education_level' in df.columns:
                valid_levels = ['Bachillerato', 'Universidad', 'Posgrado']
                invalid_levels = df[~df['education_level'].isin(valid_levels)]
                if len(invalid_levels) > 0:
                    data_issues.append(f'Niveles educativos inválidos: {len(invalid_levels)} registros')
            
            if 'main_platform' in df.columns:
                valid_platforms = ['Instagram', 'TikTok', 'Facebook', 'Twitter', 'YouTube', 'LinkedIn']
                invalid_platforms = df[~df['main_platform'].isin(valid_platforms)]
                if len(invalid_platforms) > 0:
                    data_issues.append(f'Plataformas inválidas: {len(invalid_platforms)} registros')
            
            # Verificar emails únicos
            if 'email' in df.columns:
                duplicate_emails = df['email'].duplicated().sum()
                if duplicate_emails > 0:
                    data_issues.append(f'Emails duplicados: {duplicate_emails} registros')
            
            validation['data_issues'] = data_issues
            
            if data_issues:
                validation['is_valid'] = False
        
        return validation
    
    @staticmethod
    def validate_models_excel(df: pd.DataFrame) -> Dict[str, Any]:
        """Validar específicamente archivo de modelos"""
        required_columns = [
            'id', 'name', 'description', 'category', 'difficulty',
            'is_locked', 'unlock_condition', 'estimated_time', 'use_cases'
        ]
        
        validation = ExcelValidator.validate_excel_structure(df, required_columns)
        
        if not validation['missing_columns']:
            data_issues = validation['data_issues']
            
            # Validaciones específicas de modelos
            if 'category' in df.columns:
                valid_categories = ['supervised', 'unsupervised', 'ensemble']
                invalid_categories = df[~df['category'].isin(valid_categories)]
                if len(invalid_categories) > 0:
                    data_issues.append(f'Categorías inválidas: {len(invalid_categories)} registros')
            
            if 'difficulty' in df.columns:
                valid_difficulties = ['Principiante', 'Intermedio', 'Avanzado']
                invalid_difficulties = df[~df['difficulty'].isin(valid_difficulties)]
                if len(invalid_difficulties) > 0:
                    data_issues.append(f'Dificultades inválidas: {len(invalid_difficulties)} registros')
            
            if 'estimated_time' in df.columns:
                invalid_times = df[(df['estimated_time'] < 0) | (df['estimated_time'] > 120)]
                if len(invalid_times) > 0:
                    data_issues.append(f'Tiempos estimados inválidos: {len(invalid_times)} registros')
            
            validation['data_issues'] = data_issues
            
            if data_issues:
                validation['is_valid'] = False
        
        return validation
    
    @staticmethod
    def validate_predictions_excel(df: pd.DataFrame) -> Dict[str, Any]:
        """Validar específicamente archivo de predicciones"""
        required_columns = [
            'id', 'user_id', 'model_id', 'prediction_result', 'accuracy', 'created_at'
        ]
        
        validation = ExcelValidator.validate_excel_structure(df, required_columns)
        
        if not validation['missing_columns']:
            data_issues = validation['data_issues']
            
            # Validaciones específicas de predicciones
            if 'accuracy' in df.columns:
                # Filtrar valores nulos de accuracy (algunos modelos pueden no tener accuracy)
                accuracy_values = df['accuracy'].dropna()
                invalid_accuracy = accuracy_values[(accuracy_values < 0) | (accuracy_values > 1)]
                if len(invalid_accuracy) > 0:
                    data_issues.append(f'Valores de accuracy inválidos: {len(invalid_accuracy)} registros')
            
            validation['data_issues'] = data_issues
            
            if data_issues:
                validation['is_valid'] = False
        
        return validation


class APIValidator:
    """Validador para requests de API"""
    
    @staticmethod
    def validate_json_request(data: Any) -> Dict[str, Any]:
        """Validar que el request tenga formato JSON válido"""
        if data is None:
            return {
                'is_valid': False,
                'error': 'No se enviaron datos JSON'
            }
        
        if not isinstance(data, dict):
            return {
                'is_valid': False,
                'error': 'Los datos deben ser un objeto JSON'
            }
        
        return {
            'is_valid': True,
            'data': data
        }
    
    @staticmethod
    def validate_pagination_params(page: int = None, per_page: int = None) -> Dict[str, Any]:
        """Validar parámetros de paginación"""
        errors = []
        
        if page is not None:
            if page < 1:
                errors.append('El número de página debe ser mayor a 0')
        
        if per_page is not None:
            if per_page < 1 or per_page > 100:
                errors.append('El número de elementos por página debe estar entre 1 y 100')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'page': page or 1,
            'per_page': per_page or 20
        }
    
    @staticmethod
    def validate_model_id(model_id: int) -> bool:
        """Validar ID de modelo"""
        valid_model_ids = [1, 2, 3, 4, 5, 6]  # IDs de modelos disponibles
        return model_id in valid_model_ids
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 255) -> str:
        """Sanitizar texto de entrada"""
        if not isinstance(text, str):
            return str(text)
        
        # Remover caracteres peligrosos
        sanitized = text.strip()
        sanitized = re.sub(r'[<>"\']', '', sanitized)
        
        # Limitar longitud
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
