from flask import Blueprint, request, jsonify
from services.ml_service import MLService
from utils.excel_utils import ModelsExcelHandler, PredictionsExcelHandler
import traceback

models_bp = Blueprint('models', __name__, url_prefix='/api/models')

# Inicializar servicios
ml_service = MLService()
models_handler = ModelsExcelHandler()
predictions_handler = PredictionsExcelHandler()

@models_bp.route('/', methods=['GET'])
def get_all_models():
    """Obtener todos los modelos ML disponibles"""
    try:
        df = models_handler.read_models()
        models = df.to_dict('records')
        
        # Convertir valores boolean correctamente
        for model in models:
            model['is_locked'] = bool(model['is_locked'])
            model['use_cases_list'] = [case.strip() for case in str(model['use_cases']).split(',') if case.strip()]
        
        return jsonify({
            'models': models,
            'total': len(models)
        }), 200
        
    except Exception as e:
        print(f"Error obteniendo modelos: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@models_bp.route('/<int:model_id>', methods=['GET'])
def get_model(model_id):
    """Obtener modelo específico por ID"""
    try:
        model = models_handler.get_model_by_id(model_id)
        
        # Convertir valores boolean correctamente
        model['is_locked'] = bool(model['is_locked'])
        model['use_cases_list'] = [case.strip() for case in str(model['use_cases']).split(',') if case.strip()]
        
        return jsonify({'model': model}), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print(f"Error obteniendo modelo: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@models_bp.route('/<int:model_id>/train', methods=['POST'])
def train_model(model_id):
    """Entrenar modelo específico con datos de usuario"""
    try:
        data = request.get_json()
        user_id = data.get('user_id') if data else request.args.get('user_id', type=int)
        
        if not user_id:
            return jsonify({'error': 'user_id es requerido'}), 400
        
        # Verificar que el modelo existe
        try:
            model = models_handler.get_model_by_id(model_id)
        except ValueError:
            return jsonify({'error': f'Modelo con ID {model_id} no encontrado'}), 404
        
        # Verificar si el modelo está bloqueado
        if model.get('is_locked', False):
            return jsonify({
                'error': f'Modelo bloqueado. Condición de desbloqueo: {model.get("unlock_condition", "N/A")}'
            }), 403
        
        # Ejecutar el modelo apropiado
        try:
            if model_id == 1:  # Regresión Lineal
                results = ml_service.linear_regression_analysis(user_id)
            elif model_id == 2:  # Regresión Logística
                results = ml_service.logistic_regression_analysis(user_id)
            elif model_id == 3:  # K-Means Clustering
                results = ml_service.kmeans_clustering_analysis(user_id)
            elif model_id == 4:  # Random Forest
                results = ml_service.random_forest_analysis(user_id)
            elif model_id == 5:  # Árboles de Decisión
                results = ml_service.decision_tree_analysis(user_id)
            elif model_id == 6:  # Support Vector Machine
                results = ml_service.svm_analysis(user_id)
            else:
                return jsonify({'error': 'Modelo no implementado'}), 501
            
            if results.status == "error":
                return jsonify({
                    'status': 'error',
                    'model_id': str(model_id),
                    'user_id': str(user_id),
                    'error': results.interpretation.get('summary', 'Error desconocido')
                }), 500
            
            return jsonify(results.to_dict()), 200
            
        except Exception as e:
            print(f"Error entrenando modelo {model_id}: {str(e)}")
            print(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'model_id': str(model_id),
                'user_id': str(user_id),
                'error': f'Error ejecutando modelo: {str(e)}'
            }), 500
        
    except Exception as e:
        print(f"Error general entrenando modelo: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Error interno del servidor'}), 500


@models_bp.route('/<int:model_id>/results/<int:user_id>', methods=['GET'])
def get_model_results(model_id, user_id):
    """Obtener resultados de modelo para usuario específico"""
    try:
        # Verificar que el modelo existe
        try:
            model = models_handler.get_model_by_id(model_id)
        except ValueError:
            return jsonify({'error': f'Modelo con ID {model_id} no encontrado'}), 404
        
        # Obtener resultados desde Excel
        results = predictions_handler.get_model_results(model_id, user_id)
        
        if not results:
            return jsonify({
                'message': f'No se encontraron resultados para modelo {model_id} y usuario {user_id}',
                'results': []
            }), 404
        
        return jsonify({
            'model_id': model_id,
            'user_id': user_id,
            'model_name': model['name'],
            'results': results,
            'total_results': len(results)
        }), 200
        
    except Exception as e:
        print(f"Error obteniendo resultados: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@models_bp.route('/categories', methods=['GET'])
def get_model_categories():
    """Obtener categorías de modelos disponibles"""
    try:
        df = models_handler.read_models()
        
        categories_info = {
            'supervised': {
                'name': 'Aprendizaje Supervisado',
                'description': 'Modelos que aprenden de datos etiquetados',
                'models': []
            },
            'unsupervised': {
                'name': 'Aprendizaje No Supervisado',
                'description': 'Modelos que encuentran patrones en datos sin etiquetas',
                'models': []
            },
            'ensemble': {
                'name': 'Métodos de Ensamble',
                'description': 'Modelos que combinan múltiples algoritmos',
                'models': []
            }
        }
        
        # Agrupar modelos por categoría
        for _, model in df.iterrows():
            category = model['category']
            if category in categories_info:
                categories_info[category]['models'].append({
                    'id': model['id'],
                    'name': model['name'],
                    'difficulty': model['difficulty'],
                    'is_locked': bool(model['is_locked']),
                    'estimated_time': model['estimated_time']
                })
        
        return jsonify({'categories': categories_info}), 200
        
    except Exception as e:
        print(f"Error obteniendo categorías: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@models_bp.route('/difficulty/<difficulty_level>', methods=['GET'])
def get_models_by_difficulty(difficulty_level):
    """Obtener modelos filtrados por nivel de dificultad"""
    try:
        valid_difficulties = ['Principiante', 'Intermedio', 'Avanzado']
        if difficulty_level not in valid_difficulties:
            return jsonify({
                'error': f'Nivel de dificultad debe ser uno de: {", ".join(valid_difficulties)}'
            }), 400
        
        df = models_handler.read_models()
        filtered_models = df[df['difficulty'] == difficulty_level]
        models = filtered_models.to_dict('records')
        
        # Convertir valores boolean correctamente
        for model in models:
            model['is_locked'] = bool(model['is_locked'])
            model['use_cases_list'] = [case.strip() for case in str(model['use_cases']).split(',') if case.strip()]
        
        return jsonify({
            'difficulty_level': difficulty_level,
            'models': models,
            'total': len(models)
        }), 200
        
    except Exception as e:
        print(f"Error filtrando por dificultad: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@models_bp.route('/unlocked', methods=['GET'])
def get_unlocked_models():
    """Obtener solo modelos desbloqueados"""
    try:
        df = models_handler.read_models()
        unlocked_models = df[df['is_locked'] == False]
        models = unlocked_models.to_dict('records')
        
        # Convertir valores boolean correctamente
        for model in models:
            model['is_locked'] = bool(model['is_locked'])
            model['use_cases_list'] = [case.strip() for case in str(model['use_cases']).split(',') if case.strip()]
        
        return jsonify({
            'models': models,
            'total': len(models)
        }), 200
        
    except Exception as e:
        print(f"Error obteniendo modelos desbloqueados: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@models_bp.route('/<int:model_id>/unlock', methods=['POST'])
def unlock_model(model_id):
    """Desbloquear modelo (simular cumplimiento de condición)"""
    try:
        # Verificar que el modelo existe
        try:
            model = models_handler.get_model_by_id(model_id)
        except ValueError:
            return jsonify({'error': f'Modelo con ID {model_id} no encontrado'}), 404
        
        if not model.get('is_locked', False):
            return jsonify({
                'message': f'El modelo {model["name"]} ya está desbloqueado',
                'model': model
            }), 200
        
        # Actualizar modelo para desbloquearlo
        df = models_handler.read_models()
        df.loc[df['id'] == model_id, 'is_locked'] = False
        df.loc[df['id'] == model_id, 'unlock_condition'] = 'Desbloqueado'
        
        models_handler.write_excel_data(models_handler.FILENAME, df)
        
        # Obtener modelo actualizado
        updated_model = models_handler.get_model_by_id(model_id)
        updated_model['is_locked'] = bool(updated_model['is_locked'])
        
        return jsonify({
            'message': f'Modelo {updated_model["name"]} desbloqueado exitosamente',
            'model': updated_model
        }), 200
        
    except Exception as e:
        print(f"Error desbloqueando modelo: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@models_bp.route('/compare', methods=['POST'])
def compare_models():
    """Comparar resultados de múltiples modelos para un usuario"""
    try:
        data = request.get_json()
        user_id = data.get('user_id') if data else None
        
        if not user_id:
            return jsonify({'error': 'user_id es requerido'}), 400
        
        # Ejecutar comparación de modelos
        comparison_results = ml_service.compare_models(user_id)
        
        return jsonify(comparison_results.to_dict()), 200
        
    except Exception as e:
        print(f"Error comparando modelos: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error comparando modelos: {str(e)}'}), 500


@models_bp.route('/stats', methods=['GET'])
def get_model_stats():
    """Obtener estadísticas de modelos y uso"""
    try:
        # Estadísticas de modelos
        models_df = models_handler.read_models()
        
        # Estadísticas de predicciones
        try:
            predictions_df = predictions_handler.read_excel_data(predictions_handler.FILENAME)
        except:
            predictions_df = None
        
        stats = {
            'total_models': len(models_df),
            'unlocked_models': len(models_df[models_df['is_locked'] == False]),
            'locked_models': len(models_df[models_df['is_locked'] == True]),
            'category_distribution': models_df['category'].value_counts().to_dict(),
            'difficulty_distribution': models_df['difficulty'].value_counts().to_dict(),
            'average_estimated_time': float(models_df['estimated_time'].mean()),
        }
        
        if predictions_df is not None and len(predictions_df) > 0:
            stats.update({
                'total_predictions': len(predictions_df),
                'model_usage': predictions_df['model_id'].value_counts().to_dict(),
                'unique_users_analyzed': predictions_df['user_id'].nunique(),
                'most_used_model': int(predictions_df['model_id'].mode().iloc[0]) if len(predictions_df) > 0 else None
            })
        else:
            stats.update({
                'total_predictions': 0,
                'model_usage': {},
                'unique_users_analyzed': 0,
                'most_used_model': None
            })
        
        return jsonify({'stats': stats}), 200
        
    except Exception as e:
        print(f"Error obteniendo estadísticas de modelos: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500
