from flask import Blueprint, request, jsonify, send_file
from services.visualization_service import VisualizationService
from services.excel_service import ExcelService
import traceback
import os

visualizations_bp = Blueprint('visualizations', __name__, url_prefix='/api/visualizations')

# Inicializar servicios
viz_service = VisualizationService()
excel_service = ExcelService()

@visualizations_bp.route('/dashboard', methods=['GET'])
def get_dashboard():
    """Obtener dashboard completo con todas las visualizaciones"""
    try:
        dashboard = viz_service.generate_comprehensive_dashboard()
        return jsonify(dashboard), 200
        
    except Exception as e:
        print(f"Error generando dashboard: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error generando dashboard: {str(e)}'}), 500


@visualizations_bp.route('/age-distribution', methods=['GET'])
def get_age_distribution():
    """Obtener gráfico de distribución de edades"""
    try:
        result = viz_service.generate_age_distribution()
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error generando distribución de edades: {str(e)}")
        return jsonify({'error': f'Error generando gráfico: {str(e)}'}), 500


@visualizations_bp.route('/social-vs-performance', methods=['GET'])
def get_social_vs_performance():
    """Obtener gráfico de redes sociales vs rendimiento académico"""
    try:
        result = viz_service.generate_social_media_vs_performance()
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error generando correlación: {str(e)}")
        return jsonify({'error': f'Error generando gráfico: {str(e)}'}), 500


@visualizations_bp.route('/platforms', methods=['GET'])
def get_platform_distribution():
    """Obtener gráfico de distribución de plataformas"""
    try:
        result = viz_service.generate_platform_distribution()
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error generando distribución de plataformas: {str(e)}")
        return jsonify({'error': f'Error generando gráfico: {str(e)}'}), 500


@visualizations_bp.route('/performance-by-gender', methods=['GET'])
def get_performance_by_gender():
    """Obtener gráfico de rendimiento por género"""
    try:
        result = viz_service.generate_performance_by_gender()
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error generando rendimiento por género: {str(e)}")
        return jsonify({'error': f'Error generando gráfico: {str(e)}'}), 500


@visualizations_bp.route('/correlations', methods=['GET'])
def get_correlation_heatmap():
    """Obtener mapa de calor de correlaciones"""
    try:
        result = viz_service.generate_correlation_heatmap()
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error generando mapa de correlación: {str(e)}")
        return jsonify({'error': f'Error generando gráfico: {str(e)}'}), 500


@visualizations_bp.route('/study-hours', methods=['GET'])
def get_study_hours_analysis():
    """Obtener análisis de horas de estudio"""
    try:
        result = viz_service.generate_study_hours_analysis()
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error generando análisis de estudio: {str(e)}")
        return jsonify({'error': f'Error generando gráfico: {str(e)}'}), 500


@visualizations_bp.route('/save-static', methods=['POST'])
def save_plots_to_static():
    """Guardar todas las visualizaciones en archivos estáticos"""
    try:
        result = viz_service.save_plots_to_static()
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error guardando plots estáticos: {str(e)}")
        return jsonify({'error': f'Error guardando archivos: {str(e)}'}), 500


@visualizations_bp.route('/system-status', methods=['GET'])
def get_system_status():
    """Obtener estado del sistema y archivos Excel"""
    try:
        status = excel_service.get_system_status()
        return jsonify(status), 200
        
    except Exception as e:
        print(f"Error obteniendo estado del sistema: {str(e)}")
        return jsonify({'error': f'Error obteniendo estado: {str(e)}'}), 500


@visualizations_bp.route('/data-overview', methods=['GET'])
def get_data_overview():
    """Obtener resumen general de todos los datos"""
    try:
        overview = excel_service.get_data_overview()
        return jsonify(overview), 200
        
    except Exception as e:
        print(f"Error obteniendo resumen de datos: {str(e)}")
        return jsonify({'error': f'Error obteniendo resumen: {str(e)}'}), 500


@visualizations_bp.route('/backup', methods=['POST'])
def create_backup():
    """Crear backup de todos los archivos Excel"""
    try:
        result = excel_service.backup_all_files()
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error creando backup: {str(e)}")
        return jsonify({'error': f'Error creando backup: {str(e)}'}), 500


@visualizations_bp.route('/validate', methods=['GET'])
def validate_data():
    """Validar estructura de todos los archivos Excel"""
    try:
        validation = excel_service.validate_all_files()
        return jsonify(validation), 200
        
    except Exception as e:
        print(f"Error validando datos: {str(e)}")
        return jsonify({'error': f'Error validando archivos: {str(e)}'}), 500


@visualizations_bp.route('/export', methods=['GET'])
def export_all_data():
    """Exportar todos los datos en formato JSON"""
    try:
        export_data = excel_service.export_all_data()
        return jsonify(export_data), 200
        
    except Exception as e:
        print(f"Error exportando datos: {str(e)}")
        return jsonify({'error': f'Error exportando datos: {str(e)}'}), 500


@visualizations_bp.route('/initialize', methods=['POST'])
def initialize_excel_files():
    """Inicializar archivos Excel si no existen"""
    try:
        result = excel_service.initialize_all_files()
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error inicializando archivos: {str(e)}")
        return jsonify({'error': f'Error inicializando archivos: {str(e)}'}), 500


@visualizations_bp.route('/cleanup-backups', methods=['POST'])
def cleanup_old_backups():
    """Limpiar backups antiguos"""
    try:
        retention_days = request.args.get('retention_days', default=30, type=int)
        result = excel_service.cleanup_old_backups(retention_days)
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error limpiando backups: {str(e)}")
        return jsonify({'error': f'Error limpiando backups: {str(e)}'}), 500


@visualizations_bp.route('/charts/list', methods=['GET'])
def list_available_charts():
    """Listar gráficos disponibles"""
    charts = {
        'available_charts': [
            {
                'id': 'age-distribution',
                'name': 'Distribución de Edades',
                'description': 'Histograma de distribución de edades de usuarios',
                'endpoint': '/api/visualizations/age-distribution'
            },
            {
                'id': 'social-vs-performance',
                'name': 'Redes Sociales vs Rendimiento',
                'description': 'Correlación entre uso de redes sociales y rendimiento académico',
                'endpoint': '/api/visualizations/social-vs-performance'
            },
            {
                'id': 'platforms',
                'name': 'Distribución de Plataformas',
                'description': 'Distribución de plataformas principales de redes sociales',
                'endpoint': '/api/visualizations/platforms'
            },
            {
                'id': 'performance-by-gender',
                'name': 'Rendimiento por Género',
                'description': 'Comparación de rendimiento académico por género',
                'endpoint': '/api/visualizations/performance-by-gender'
            },
            {
                'id': 'correlations',
                'name': 'Matriz de Correlación',
                'description': 'Mapa de calor de correlaciones entre variables',
                'endpoint': '/api/visualizations/correlations'
            },
            {
                'id': 'study-hours',
                'name': 'Análisis de Horas de Estudio',
                'description': 'Análisis completo de patrones de horas de estudio',
                'endpoint': '/api/visualizations/study-hours'
            },
            {
                'id': 'dashboard',
                'name': 'Dashboard Completo',
                'description': 'Dashboard con todas las visualizaciones',
                'endpoint': '/api/visualizations/dashboard'
            }
        ],
        'total_charts': 7,
        'categories': {
            'distribution': ['age-distribution', 'platforms'],
            'correlation': ['social-vs-performance', 'correlations'],
            'comparison': ['performance-by-gender'],
            'analysis': ['study-hours'],
            'comprehensive': ['dashboard']
        }
    }
    
    return jsonify(charts), 200


@visualizations_bp.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check para visualizaciones"""
    try:
        # Verificar que los servicios estén funcionando
        status = excel_service.get_system_status()
        
        health_status = {
            'status': 'healthy' if status['health'] == 'healthy' else 'degraded',
            'timestamp': status['timestamp'],
            'services': {
                'visualization_service': 'active',
                'excel_service': 'active',
                'matplotlib': 'available',
                'seaborn': 'available'
            },
            'files_status': status['files'],
            'message': 'Servicio de visualizaciones funcionando correctamente'
        }
        
        return jsonify(health_status), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'message': 'Error en servicio de visualizaciones'
        }), 500
