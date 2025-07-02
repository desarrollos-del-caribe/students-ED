from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from services.analysis_model import (
   social_media_addiction_risk, academic_performance_risk,
    student_performance_prediction , addiction_by_country
)
from config import Config

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["500 per day", "100 per hour"],
    storage_uri="memory://"
)

@app.route('/api')
def home():
    return jsonify({"message": "API funcionando correctamente"})


@app.route('/students/<history_model>', methods=['GET'])
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


@app.route('/social_media_addiction_risk', methods=['POST'])
def addiction_risk():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json"}), 415
        data = request.get_json() or {}
        history_id = data.get('history_models_import_id')
        result = social_media_addiction_risk(
            float(data.get('usage_hours', 0)),
            float(data.get('addicted_score', 0)),
            float(data.get('mental_health_score', 0)),
            float(data.get('conflicts_score', 0)),
            history_id
        )
        return jsonify({"risk": result})
    except ValueError as e:
        return jsonify({"error": f"Formato de parámetros inválido: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error al predecir riesgo de adicción: {str(e)}"}), 500


@app.route('/academic_perfomance_risk', methods=['POST'])
def academic_risk():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json"}), 415
        data = request.get_json() or {}
        history_id = data.get('history_models_import_id')
        result = academic_performance_risk(
            float(data.get('usage_hours', 0)),
            float(data.get('sleep_hours', 0)),
            float(data.get('mental_health_score', 0)),
            history_id
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": f"Formato de parámetros inválido: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error al predecir riesgo académico: {str(e)}"}), 500


@app.route('/student_performance_prediction', methods=['POST'])
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
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": f"Formato de id inválido: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error al predecir rendimiento del estudiante: {str(e)}"}), 500

@app.route('/addiction_by_country', methods=['POST'])
def country_addiction():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json"}), 415
        data = request.get_json() or {}
        history_id = data.get('history_models_import_id')
        result = addiction_by_country(history_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error al calcular adicción por país: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='*', port=5000, debug=False)
