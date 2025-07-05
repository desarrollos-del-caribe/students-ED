import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

from utils.excel_utils import UsersExcelHandler, ModelsExcelHandler, PredictionsExcelHandler
from utils.validators import ExcelValidator
from utils.helpers import format_error_response, create_summary_statistics

class ExcelService:
    """Servicio centralizado para operaciones con Excel"""
    
    def __init__(self):
        self.users_handler = UsersExcelHandler()
        self.models_handler = ModelsExcelHandler()
        self.predictions_handler = PredictionsExcelHandler()
    
    def initialize_all_files(self) -> Dict[str, Any]:
        """Inicializar todos los archivos Excel"""
        try:
            results = {
                'users': False,
                'models': False,
                'predictions': False,
                'errors': []
            }
            
            # Inicializar usuarios
            try:
                if not self.users_handler.file_exists(self.users_handler.FILENAME):
                    self.users_handler.initialize_if_not_exists()
                results['users'] = True
            except Exception as e:
                results['errors'].append(f"Error inicializando usuarios: {str(e)}")
            
            # Inicializar modelos
            try:
                if not self.models_handler.file_exists(self.models_handler.FILENAME):
                    self.models_handler.initialize_if_not_exists()
                results['models'] = True
            except Exception as e:
                results['errors'].append(f"Error inicializando modelos: {str(e)}")
            
            # Inicializar predicciones
            try:
                if not self.predictions_handler.file_exists(self.predictions_handler.FILENAME):
                    self.predictions_handler.initialize_if_not_exists()
                results['predictions'] = True
            except Exception as e:
                results['errors'].append(f"Error inicializando predicciones: {str(e)}")
            
            return results
            
        except Exception as e:
            return {
                'users': False,
                'models': False,
                'predictions': False,
                'errors': [f"Error general: {str(e)}"]
            }
    
    def validate_all_files(self) -> Dict[str, Any]:
        """Validar estructura de todos los archivos Excel"""
        validation_results = {}
        
        # Validar usuarios
        try:
            users_df = self.users_handler.read_users()
            validation_results['users'] = ExcelValidator.validate_users_excel(users_df)
        except Exception as e:
            validation_results['users'] = {
                'is_valid': False,
                'error': str(e)
            }
        
        # Validar modelos
        try:
            models_df = self.models_handler.read_models()
            validation_results['models'] = ExcelValidator.validate_models_excel(models_df)
        except Exception as e:
            validation_results['models'] = {
                'is_valid': False,
                'error': str(e)
            }
        
        # Validar predicciones
        try:
            if self.predictions_handler.file_exists(self.predictions_handler.FILENAME):
                predictions_df = self.predictions_handler.read_excel_data(self.predictions_handler.FILENAME)
                validation_results['predictions'] = ExcelValidator.validate_predictions_excel(predictions_df)
            else:
                validation_results['predictions'] = {
                    'is_valid': True,
                    'message': 'Archivo de predicciones no existe (normal en primera ejecución)'
                }
        except Exception as e:
            validation_results['predictions'] = {
                'is_valid': False,
                'error': str(e)
            }
        
        return validation_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado general del sistema"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'files': {},
            'statistics': {},
            'validation': {},
            'health': 'unknown'
        }
        
        try:
            # Estado de archivos
            status['files'] = {
                'users': {
                    'exists': self.users_handler.file_exists(self.users_handler.FILENAME),
                    'path': self.users_handler.get_file_path(self.users_handler.FILENAME),
                    'size_mb': self._get_file_size_mb(self.users_handler.get_file_path(self.users_handler.FILENAME))
                },
                'models': {
                    'exists': self.models_handler.file_exists(self.models_handler.FILENAME),
                    'path': self.models_handler.get_file_path(self.models_handler.FILENAME),
                    'size_mb': self._get_file_size_mb(self.models_handler.get_file_path(self.models_handler.FILENAME))
                },
                'predictions': {
                    'exists': self.predictions_handler.file_exists(self.predictions_handler.FILENAME),
                    'path': self.predictions_handler.get_file_path(self.predictions_handler.FILENAME),
                    'size_mb': self._get_file_size_mb(self.predictions_handler.get_file_path(self.predictions_handler.FILENAME))
                }
            }
            
            # Estadísticas
            if status['files']['users']['exists']:
                users_df = self.users_handler.read_users()
                status['statistics']['users'] = create_summary_statistics(users_df)
            
            if status['files']['models']['exists']:
                models_df = self.models_handler.read_models()
                status['statistics']['models'] = create_summary_statistics(models_df)
            
            if status['files']['predictions']['exists']:
                predictions_df = self.predictions_handler.read_excel_data(self.predictions_handler.FILENAME)
                status['statistics']['predictions'] = create_summary_statistics(predictions_df)
            
            # Validaciones
            status['validation'] = self.validate_all_files()
            
            # Estado de salud
            all_files_exist = all(file_info['exists'] for file_info in status['files'].values())
            all_validations_ok = all(
                validation.get('is_valid', False) 
                for validation in status['validation'].values() 
                if 'is_valid' in validation
            )
            
            if all_files_exist and all_validations_ok:
                status['health'] = 'healthy'
            elif all_files_exist:
                status['health'] = 'warning'
            else:
                status['health'] = 'error'
            
        except Exception as e:
            status['health'] = 'error'
            status['error'] = str(e)
        
        return status
    
    def backup_all_files(self) -> Dict[str, Any]:
        """Crear backup de todos los archivos Excel"""
        backup_results = {
            'timestamp': datetime.now().isoformat(),
            'backups': {},
            'success': True,
            'errors': []
        }
        
        try:
            # Backup usuarios
            if self.users_handler.file_exists(self.users_handler.FILENAME):
                backup_path = self.users_handler.backup_excel_file(self.users_handler.FILENAME)
                backup_results['backups']['users'] = backup_path
            
            # Backup modelos
            if self.models_handler.file_exists(self.models_handler.FILENAME):
                backup_path = self.models_handler.backup_excel_file(self.models_handler.FILENAME)
                backup_results['backups']['models'] = backup_path
            
            # Backup predicciones
            if self.predictions_handler.file_exists(self.predictions_handler.FILENAME):
                backup_path = self.predictions_handler.backup_excel_file(self.predictions_handler.FILENAME)
                backup_results['backups']['predictions'] = backup_path
            
        except Exception as e:
            backup_results['success'] = False
            backup_results['errors'].append(str(e))
        
        return backup_results
    
    def get_data_overview(self) -> Dict[str, Any]:
        """Obtener resumen general de todos los datos"""
        overview = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'correlations': {},
            'insights': []
        }
        
        try:
            # Datos de usuarios
            if self.users_handler.file_exists(self.users_handler.FILENAME):
                users_df = self.users_handler.read_users()
                
                overview['summary']['users'] = {
                    'total': len(users_df),
                    'average_age': float(users_df['age'].mean()) if 'age' in users_df.columns else 0,
                    'average_social_media_usage': float(users_df['social_media_usage'].mean()) if 'social_media_usage' in users_df.columns else 0,
                    'average_academic_performance': float(users_df['academic_performance'].mean()) if 'academic_performance' in users_df.columns else 0,
                    'gender_distribution': users_df['gender'].value_counts().to_dict() if 'gender' in users_df.columns else {},
                    'platform_distribution': users_df['main_platform'].value_counts().to_dict() if 'main_platform' in users_df.columns else {}
                }
                
                # Correlaciones
                if len(users_df) > 1:
                    numeric_cols = ['age', 'social_media_usage', 'academic_performance', 'study_hours']
                    available_cols = [col for col in numeric_cols if col in users_df.columns]
                    
                    if len(available_cols) >= 2:
                        correlation_matrix = users_df[available_cols].corr()
                        overview['correlations'] = correlation_matrix.to_dict()
                
                # Insights
                if 'social_media_usage' in users_df.columns and 'academic_performance' in users_df.columns:
                    correlation = users_df['social_media_usage'].corr(users_df['academic_performance'])
                    if correlation < -0.3:
                        overview['insights'].append("Correlación negativa fuerte entre uso de redes sociales y rendimiento académico")
                    elif correlation < -0.1:
                        overview['insights'].append("Correlación negativa moderada entre uso de redes sociales y rendimiento académico")
            
            # Datos de modelos
            if self.models_handler.file_exists(self.models_handler.FILENAME):
                models_df = self.models_handler.read_models()
                
                overview['summary']['models'] = {
                    'total': len(models_df),
                    'unlocked': len(models_df[models_df['is_locked'] == False]) if 'is_locked' in models_df.columns else 0,
                    'by_category': models_df['category'].value_counts().to_dict() if 'category' in models_df.columns else {},
                    'by_difficulty': models_df['difficulty'].value_counts().to_dict() if 'difficulty' in models_df.columns else {}
                }
            
            # Datos de predicciones
            if self.predictions_handler.file_exists(self.predictions_handler.FILENAME):
                predictions_df = self.predictions_handler.read_excel_data(self.predictions_handler.FILENAME)
                
                overview['summary']['predictions'] = {
                    'total': len(predictions_df),
                    'unique_users': predictions_df['user_id'].nunique() if 'user_id' in predictions_df.columns else 0,
                    'models_used': predictions_df['model_id'].value_counts().to_dict() if 'model_id' in predictions_df.columns else {},
                    'average_accuracy': float(predictions_df['accuracy'].mean()) if 'accuracy' in predictions_df.columns and not predictions_df['accuracy'].isna().all() else None
                }
        
        except Exception as e:
            overview['error'] = str(e)
        
        return overview
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Obtener tamaño de archivo en MB"""
        try:
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                return round(size_bytes / (1024 * 1024), 2)
        except:
            pass
        return 0.0
    
    def export_all_data(self) -> Dict[str, Any]:
        """Exportar todos los datos en formato JSON"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'data': {}
        }
        
        try:
            # Exportar usuarios
            if self.users_handler.file_exists(self.users_handler.FILENAME):
                users_df = self.users_handler.read_users()
                export_data['data']['users'] = users_df.to_dict('records')
            
            # Exportar modelos
            if self.models_handler.file_exists(self.models_handler.FILENAME):
                models_df = self.models_handler.read_models()
                export_data['data']['models'] = models_df.to_dict('records')
            
            # Exportar predicciones
            if self.predictions_handler.file_exists(self.predictions_handler.FILENAME):
                predictions_df = self.predictions_handler.read_excel_data(self.predictions_handler.FILENAME)
                export_data['data']['predictions'] = predictions_df.to_dict('records')
            
        except Exception as e:
            export_data['error'] = str(e)
        
        return export_data
    
    def cleanup_old_backups(self, retention_days: int = 30) -> Dict[str, Any]:
        """Limpiar backups antiguos"""
        cleanup_results = {
            'timestamp': datetime.now().isoformat(),
            'files_removed': [],
            'files_kept': [],
            'errors': []
        }
        
        try:
            backup_dir = self.users_handler.backup_path
            
            if os.path.exists(backup_dir):
                cutoff_date = datetime.now().timestamp() - (retention_days * 24 * 60 * 60)
                
                for filename in os.listdir(backup_dir):
                    file_path = os.path.join(backup_dir, filename)
                    
                    if os.path.isfile(file_path):
                        file_modified_time = os.path.getmtime(file_path)
                        
                        if file_modified_time < cutoff_date:
                            try:
                                os.remove(file_path)
                                cleanup_results['files_removed'].append(filename)
                            except Exception as e:
                                cleanup_results['errors'].append(f"Error eliminando {filename}: {str(e)}")
                        else:
                            cleanup_results['files_kept'].append(filename)
        
        except Exception as e:
            cleanup_results['errors'].append(f"Error general: {str(e)}")
        
        return cleanup_results
