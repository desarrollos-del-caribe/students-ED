from flask import Blueprint, request, jsonify
from services.ml_service import MLService
from utils.excel_utils import UsersExcelHandler, PredictionsExcelHandler
import traceback

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api')

# Inicializar servicios
ml_service = MLService()
users_handler = UsersExcelHandler()
predictions_handler = PredictionsExcelHandler()

@analysis_bp.route('/analyze/user/<int:user_id>', methods=['POST'])
def analyze_user(user_id):
    """Análisis completo de usuario con todos los modelos disponibles"""
    try:
        # Verificar que el usuario existe
        try:
            user = users_handler.get_user_by_id(user_id)
        except ValueError:
            return jsonify({'error': f'Usuario con ID {user_id} no encontrado'}), 404
        
        # Ejecutar análisis completo
        analysis_results = ml_service.compare_models(user_id)
        
        # Ejecutar modelos individuales para obtener resultados detallados
        individual_results = {}
        
        try:
            # Regresión Lineal (modelo 1)
            linear_result = ml_service.linear_regression_analysis(user_id)
            individual_results['linear_regression'] = linear_result.to_dict()
        except Exception as e:
            individual_results['linear_regression'] = {'error': str(e)}
        
        try:
            # Regresión Logística (modelo 2)
            logistic_result = ml_service.logistic_regression_analysis(user_id)
            individual_results['logistic_regression'] = logistic_result.to_dict()
        except Exception as e:
            individual_results['logistic_regression'] = {'error': str(e)}
        
        try:
            # K-Means Clustering (modelo 3)
            kmeans_result = ml_service.kmeans_clustering_analysis(user_id)
            individual_results['kmeans_clustering'] = kmeans_result.to_dict()
        except Exception as e:
            individual_results['kmeans_clustering'] = {'error': str(e)}
        
        try:
            # Random Forest (modelo 4)
            rf_result = ml_service.random_forest_analysis(user_id)
            individual_results['random_forest'] = rf_result.to_dict()
        except Exception as e:
            individual_results['random_forest'] = {'error': str(e)}
        
        try:
            # Árboles de Decisión (modelo 5)
            dt_result = ml_service.decision_tree_analysis(user_id)
            individual_results['decision_tree'] = dt_result.to_dict()
        except Exception as e:
            individual_results['decision_tree'] = {'error': str(e)}
        
        try:
            # Support Vector Machine (modelo 6)
            svm_result = ml_service.svm_analysis(user_id)
            individual_results['support_vector_machine'] = svm_result.to_dict()
        except Exception as e:
            individual_results['support_vector_machine'] = {'error': str(e)}
        
        # Compilar resultado final
        final_result = {
            'user_info': user,
            'comprehensive_analysis': analysis_results.to_dict(),
            'individual_models': individual_results,
            'summary': {
                'total_models_executed': len([r for r in individual_results.values() if 'error' not in r]),
                'models_with_errors': len([r for r in individual_results.values() if 'error' in r]),
                'analysis_timestamp': individual_results.get('linear_regression', {}).get('timestamp', ''),
                'risk_assessment': analysis_results.user_profile.risk_level if analysis_results.user_profile else 'No evaluado',
                'academic_prediction': analysis_results.user_profile.academic_prediction if analysis_results.user_profile else 0
            }
        }
        
        return jsonify(final_result), 200
        
    except Exception as e:
        print(f"Error en análisis completo: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error en análisis completo: {str(e)}'}), 500


@analysis_bp.route('/predictions/<int:user_id>', methods=['GET'])
def get_user_predictions(user_id):
    """Obtener todas las predicciones de un usuario"""
    try:
        # Verificar que el usuario existe
        try:
            user = users_handler.get_user_by_id(user_id)
        except ValueError:
            return jsonify({'error': f'Usuario con ID {user_id} no encontrado'}), 404
        
        # Obtener predicciones desde Excel
        predictions = predictions_handler.get_user_predictions(user_id)
        
        if not predictions:
            return jsonify({
                'user_id': user_id,
                'user_name': user['name'],
                'predictions': [],
                'total_predictions': 0,
                'message': 'No se encontraron predicciones para este usuario'
            }), 200
        
        # Enriquecer predicciones con información del modelo
        from utils.excel_utils import ModelsExcelHandler
        models_handler = ModelsExcelHandler()
        
        enriched_predictions = []
        for prediction in predictions:
            try:
                model = models_handler.get_model_by_id(prediction['model_id'])
                enriched_prediction = {
                    **prediction,
                    'model_name': model['name'],
                    'model_category': model['category'],
                    'model_difficulty': model['difficulty']
                }
                enriched_predictions.append(enriched_prediction)
            except:
                enriched_predictions.append(prediction)
        
        # Agrupar por modelo
        predictions_by_model = {}
        for pred in enriched_predictions:
            model_id = pred['model_id']
            if model_id not in predictions_by_model:
                predictions_by_model[model_id] = {
                    'model_id': model_id,
                    'model_name': pred.get('model_name', f'Modelo {model_id}'),
                    'predictions': []
                }
            predictions_by_model[model_id]['predictions'].append(pred)
        
        return jsonify({
            'user_id': user_id,
            'user_name': user['name'],
            'predictions': enriched_predictions,
            'predictions_by_model': list(predictions_by_model.values()),
            'total_predictions': len(predictions),
            'models_used': len(predictions_by_model),
            'latest_prediction': max(enriched_predictions, key=lambda x: x['created_at']) if enriched_predictions else None
        }), 200
        
    except Exception as e:
        print(f"Error obteniendo predicciones: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@analysis_bp.route('/predictions/all', methods=['GET'])
def get_all_predictions():
    """Obtener todas las predicciones con filtros opcionales"""
    try:
        # Parámetros de filtro
        model_id = request.args.get('model_id', type=int)
        user_id = request.args.get('user_id', type=int)
        limit = request.args.get('limit', default=100, type=int)
        
        try:
            df = predictions_handler.read_excel_data(predictions_handler.FILENAME)
        except:
            return jsonify({
                'predictions': [],
                'total': 0,
                'message': 'No hay predicciones registradas'
            }), 200
        
        # Aplicar filtros
        if model_id:
            df = df[df['model_id'] == model_id]
        
        if user_id:
            df = df[df['user_id'] == user_id]
        
        # Limitar resultados
        df = df.tail(limit)
        
        predictions = df.to_dict('records')
        
        # Enriquecer con información adicional
        from utils.excel_utils import ModelsExcelHandler
        models_handler = ModelsExcelHandler()
        
        enriched_predictions = []
        for prediction in predictions:
            try:
                model = models_handler.get_model_by_id(prediction['model_id'])
                user = users_handler.get_user_by_id(prediction['user_id'])
                
                enriched_prediction = {
                    **prediction,
                    'model_name': model['name'],
                    'user_name': user['name']
                }
                enriched_predictions.append(enriched_prediction)
            except:
                enriched_predictions.append(prediction)
        
        return jsonify({
            'predictions': enriched_predictions,
            'total': len(enriched_predictions),
            'filters': {
                'model_id': model_id,
                'user_id': user_id,
                'limit': limit
            }
        }), 200
        
    except Exception as e:
        print(f"Error obteniendo todas las predicciones: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@analysis_bp.route('/recommendations/<int:user_id>', methods=['GET'])
def get_user_recommendations(user_id):
    """Obtener recomendaciones personalizadas para un usuario"""
    try:
        # Verificar que el usuario exists
        try:
            user = users_handler.get_user_by_id(user_id)
        except ValueError:
            return jsonify({'error': f'Usuario con ID {user_id} no encontrado'}), 404
        
        # Obtener datos del usuario
        social_media_usage = user['social_media_usage']
        academic_performance = user['academic_performance']
        study_hours = user['study_hours']
        main_platform = user['main_platform']
        
        # Generar recomendaciones basadas en datos
        recommendations = {
            'immediate_actions': [],
            'long_term_strategies': [],
            'resource_suggestions': []
        }
        
        # Recomendaciones inmediatas
        if social_media_usage > 7:
            recommendations['immediate_actions'].extend([
                'Reducir el tiempo diario en redes sociales',
                'Establecer horarios específicos para redes sociales',
                'Usar aplicaciones de control de tiempo de pantalla'
            ])
        
        if study_hours < 20:
            recommendations['immediate_actions'].extend([
                'Incrementar las horas de estudio semanales',
                'Crear un horario de estudio estructurado',
                'Eliminar distracciones durante el tiempo de estudio'
            ])
        
        if academic_performance < 75:
            recommendations['immediate_actions'].extend([
                'Buscar apoyo académico adicional',
                'Revisar métodos de estudio actuales',
                'Consultar con profesores sobre áreas de mejora'
            ])
        
        # Estrategias a largo plazo
        if social_media_usage > 5:
            recommendations['long_term_strategies'].extend([
                'Desarrollar hábitos digitales saludables',
                'Implementar períodos de desconexión digital',
                'Usar redes sociales de manera productiva'
            ])
        
        recommendations['long_term_strategies'].extend([
            'Establecer metas académicas claras y medibles',
            'Desarrollar técnicas de gestión del tiempo',
            'Crear un equilibrio entre vida digital y académica'
        ])
        
        # Sugerencias de recursos
        recommendations['resource_suggestions'].extend([
            'Apps de productividad: Forest, Pomodoro Timer',
            'Plataformas educativas: Khan Academy, Coursera',
            'Técnicas de estudio: Método Cornell, Mapas mentales'
        ])
        
        # Recomendaciones específicas por plataforma
        platform_recommendations = {
            'Instagram': ['Seguir cuentas educativas', 'Limitar tiempo en Stories'],
            'TikTok': ['Usar TikTok para contenido educativo', 'Evitar uso durante horas de estudio'],
            'YouTube': ['Crear listas de reproducción educativas', 'Usar temporizadores'],
            'LinkedIn': ['Conectar con profesionales del área de estudio', 'Participar en grupos académicos'],
            'Facebook': ['Unirse a grupos de estudio', 'Limitar notificaciones'],
            'Twitter': ['Seguir cuentas académicas relevantes', 'Usar listas temáticas']
        }
        
        if main_platform in platform_recommendations:
            recommendations['resource_suggestions'].extend(platform_recommendations[main_platform])
        
        # Análisis de riesgo
        risk_factors = []
        if social_media_usage > 6:
            risk_factors.append('Alto uso de redes sociales')
        if study_hours < 25:
            risk_factors.append('Pocas horas de estudio')
        if academic_performance < 70:
            risk_factors.append('Rendimiento académico bajo')
        
        risk_level = 'Alto' if len(risk_factors) >= 2 else 'Medio' if len(risk_factors) == 1 else 'Bajo'
        
        return jsonify({
            'user_id': user_id,
            'user_name': user['name'],
            'risk_assessment': {
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'protective_factors': [
                    f'Horas de estudio: {study_hours}',
                    f'Rendimiento actual: {academic_performance}',
                    f'Plataforma principal: {main_platform}'
                ]
            },
            'recommendations': recommendations,
            'personalized_tips': [
                f'Considera reducir el uso de {main_platform} en {max(0, social_media_usage - 5)} horas',
                f'Incrementa las horas de estudio en {max(0, 30 - study_hours)} horas semanales',
                f'Tu rendimiento actual de {academic_performance}% {"necesita mejora" if academic_performance < 75 else "está en buen nivel"}'
            ]
        }), 200
        
    except Exception as e:
        print(f"Error generando recomendaciones: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@analysis_bp.route('/trends', methods=['GET'])
def get_analysis_trends():
    """Obtener tendencias de análisis y predicciones"""
    try:
        # Leer datos de usuarios y predicciones
        users_df = users_handler.read_users()
        
        try:
            predictions_df = predictions_handler.read_excel_data(predictions_handler.FILENAME)
        except:
            predictions_df = None
        
        trends = {
            'user_trends': {
                'total_users': len(users_df),
                'average_social_media_usage': float(users_df['social_media_usage'].mean()),
                'average_academic_performance': float(users_df['academic_performance'].mean()),
                'platform_popularity': users_df['main_platform'].value_counts().to_dict(),
                'performance_distribution': {
                    'excellent': len(users_df[users_df['academic_performance'] >= 90]),
                    'good': len(users_df[(users_df['academic_performance'] >= 75) & (users_df['academic_performance'] < 90)]),
                    'average': len(users_df[(users_df['academic_performance'] >= 60) & (users_df['academic_performance'] < 75)]),
                    'poor': len(users_df[users_df['academic_performance'] < 60])
                }
            }
        }
        
        if predictions_df is not None and len(predictions_df) > 0:
            trends['prediction_trends'] = {
                'total_predictions': len(predictions_df),
                'models_used': predictions_df['model_id'].value_counts().to_dict(),
                'users_analyzed': predictions_df['user_id'].nunique(),
                'average_accuracy': float(predictions_df['accuracy'].mean()) if 'accuracy' in predictions_df.columns else None
            }
        else:
            trends['prediction_trends'] = {
                'total_predictions': 0,
                'models_used': {},
                'users_analyzed': 0,
                'average_accuracy': None
            }
        
        # Correlaciones
        correlations = {
            'social_media_vs_performance': float(users_df['social_media_usage'].corr(users_df['academic_performance'])),
            'study_hours_vs_performance': float(users_df['study_hours'].corr(users_df['academic_performance'])),
            'age_vs_social_media': float(users_df['age'].corr(users_df['social_media_usage']))
        }
        
        trends['correlations'] = correlations
        
        return jsonify({'trends': trends}), 200
        
    except Exception as e:
        print(f"Error obteniendo tendencias: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@analysis_bp.route('/export/predictions', methods=['GET'])
def export_predictions():
    """Exportar predicciones en formato JSON para backup"""
    try:
        user_id = request.args.get('user_id', type=int)
        model_id = request.args.get('model_id', type=int)
        
        try:
            df = predictions_handler.read_excel_data(predictions_handler.FILENAME)
        except:
            return jsonify({
                'predictions': [],
                'total': 0,
                'message': 'No hay predicciones para exportar'
            }), 200
        
        # Aplicar filtros si se especifican
        if user_id:
            df = df[df['user_id'] == user_id]
        
        if model_id:
            df = df[df['model_id'] == model_id]
        
        predictions = df.to_dict('records')
        
        return jsonify({
            'export_timestamp': predictions_handler.read_excel_data(predictions_handler.FILENAME).index.max() if len(df) > 0 else None,
            'filters_applied': {
                'user_id': user_id,
                'model_id': model_id
            },
            'total_exported': len(predictions),
            'predictions': predictions
        }), 200
        
    except Exception as e:
        print(f"Error exportando predicciones: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500
