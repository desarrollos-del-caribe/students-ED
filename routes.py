from flask import Blueprint, request, jsonify
from services.analysis_model import (
    social_media_addiction_risk, academic_performance_risk,
    student_performance_prediction, addiction_by_country
)
from config import Config

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API funcionando correctamente"})

@api_bp.route('/students/<history_model>', methods=['GET'])
def get_students(history_model):
    try:
        conn = Config.get_connection()
        if conn is None:
            raise Exception("No se pudo establecer la conexión a la base de datos")
        cursor = conn.cursor(as_dict=True)

        if history_model.lower() == 'null':
            query = """
                SELECT id, age, gender_id, academic_level_id, country_id, avg_daily_used_hours,
                       social_network_id, affects_academic_performance, sleep_hours_per_night,
                       mental_health_score, relationship_status_id, conflicts_over_social_media,
                       addicted_score
                FROM Tbl_Students_Model
            """
            params = ()
        else:
            query = """
                SELECT id, age, gender_id, academic_level_id, country_id, avg_daily_used_hours,
                       social_network_id, affects_academic_performance, sleep_hours_per_night,
                       mental_health_score, relationship_status_id, conflicts_over_social_media,
                       addicted_score
                FROM Tbl_Students_Model
                WHERE history_models_import_id = %s
            """
            params = (history_model,)

        cursor.execute(query, params)
        students = cursor.fetchall()
        conn.close()
        return jsonify({'students': students}), 200
    except Exception as e:
        if 'conn' in locals() and conn is not None:
            conn.close()
        return jsonify({'error': f"Error al obtener estudiantes: {str(e)}"}), 500

@api_bp.route('/social_media_addiction_risk', methods=['POST'])
def addiction_risk():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json"}), 415
        data = request.get_json() or {}
        history_id = data.get('history_models_import_id')
        required_params = ['usage_hours', 'addicted_score', 'mental_health_score', 'conflicts_score']
        if not all(param in data for param in required_params):
            return jsonify({"error": "Faltan parámetros requeridos"}), 400
        result = social_media_addiction_risk(
            float(data.get('usage_hours', 0)),
            float(data.get('addicted_score', 0)),
            float(data.get('mental_health_score', 0)),
            float(data.get('conflicts_score', 0)),
            history_id
        )
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": f"Formato de parámetros inválido: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error al predecir riesgo de adicción: {str(e)}"}), 500

@api_bp.route('/academic_performance_risk', methods=['POST'])
def academic_risk():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json"}), 415
        data = request.get_json() or {}
        history_id = data.get('history_models_import_id')
        required_params = ['usage_hours', 'sleep_hours', 'mental_health_score']
        if not all(param in data for param in required_params):
            return jsonify({"error": "Faltan parámetros requeridos"}), 400
        result = academic_performance_risk(
            float(data.get('usage_hours', 0)),
            float(data.get('sleep_hours', 0)),
            float(data.get('mental_health_score', 0)),
            history_id
        )
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": f"Formato de parámetros inválido: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error al predecir riesgo académico: {str(e)}"}), 500

@api_bp.route('/student_performance_prediction', methods=['POST'])
def student_pred():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json"}), 415
        data = request.get_json() or {}
        history_id = data.get('history_models_import_id')
        student_id = data.get('id')
        if student_id is None:
            return jsonify({"error": "id es requerido"}), 400
        result = student_performance_prediction(int(student_id), history_id)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": f"Formato de id inválido: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error al predecir rendimiento del estudiante: {str(e)}"}), 500

@api_bp.route('/addiction_by_country', methods=['POST'])
def country_addiction():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json"}), 415
        data = request.get_json() or {}
        history_id = data.get('history_models_import_id')
        min_students = data.get('min_students', 5)
        result = addiction_by_country(history_id, min_students)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Error al calcular adicción por país: {str(e)}"}), 500