from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from services.data_processor import process_and_insert_data
from services.analysis_model import (
    social_media_addiction_risk,
    academic_performance_risk,
    sleep_prediction,
    get_platform_distribution
)
import pymssql
from config import Config
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["500 per day", "100 per hour"],
    storage_uri="memory://"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.before_request
def restrict_access():
    allowed_ips = ['127.0.0.1']
    if request.remote_addr not in allowed_ips:
        logger.warning(f"Acceso denegado desde {request.remote_addr} con User-Agent: {request.headers.get('User-Agent')}")
        return jsonify({'error': 'Acceso no autorizado'}), 403

@app.route('/upload', methods=['POST', 'OPTIONS'])
@limiter.limit("20 per minute")
def upload_file():
    logger.info(f"Solicitud recibida: {request.method} {request.path} from {request.remote_addr} con User-Agent: {request.headers.get('User-Agent')}")
    if request.method == 'OPTIONS':
        return '', 200
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó archivo'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    result = process_and_insert_data(file)
    status_code = 200 if result.get('success') else 409
    return jsonify(result), status_code

@app.route('/students', methods=['GET', 'OPTIONS'])
@limiter.limit("20 per minute")
def get_students():
    logger.info(f"Solicitud recibida: {request.method} {request.path} from {request.remote_addr} con User-Agent: {request.headers.get('User-Agent')}")
    if request.method == 'OPTIONS':
        return '', 200
    try:
        conn = Config.get_connection()
        cursor = conn.cursor(as_dict=True)
        cursor.execute("""
            SELECT id, age, gender_id, academic_level_id, country_id, avg_daily_used_hours,
                   social_network_id, affects_academic_performance, sleep_hours_per_night,
                   mental_health_score, relationship_status_id, conflicts_over_social_media,
                   addicted_score
            FROM Tbl_Students_Model WHERE history_models_import_id = 1
        """)
        students = cursor.fetchall()
        conn.close()
        return jsonify({'students': students}), 200
    except Exception as e:
        logger.error(f"Error obteniendo estudiantes: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/student/<int:student_id>', methods=['GET', 'OPTIONS'])
@limiter.limit("20 per minute")
def get_student(student_id):
    logger.info(f"Solicitud recibida: {request.method} {request.path} from {request.remote_addr} con User-Agent: {request.headers.get('User-Agent')}")
    if request.method == 'OPTIONS':
        return '', 200
    try:
        conn = Config.get_connection()
        cursor = conn.cursor(as_dict=True)
        cursor.execute("""
            SELECT id, age, gender_id, academic_level_id, country_id, avg_daily_used_hours,
                   social_network_id, affects_academic_performance, sleep_hours_per_night,
                   mental_health_score, relationship_status_id, conflicts_over_social_media,
                   addicted_score
            FROM Tbl_Students_Model
            WHERE id = %s
        """, (student_id,))
        student = cursor.fetchone()
        conn.close()
        if not student:
            return jsonify({'error': 'Estudiante no encontrado'}), 404
        return jsonify({'student': student}), 200
    except Exception as e:
        logger.error(f"Error obteniendo estudiante {student_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/addiction', methods=['POST', 'OPTIONS'])
@limiter.limit("50 per minute")
def predict_addiction():
    logger.info(f"Solicitud recibida: {request.method} {request.path} from {request.remote_addr} con User-Agent: {request.headers.get('User-Agent')}")
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    result = social_media_addiction_risk(
        data.get('usage_hours', 0),
        data.get('addicted_score', 0),
        data.get('mental_health_score', 0),
        data.get('conflicts_score', 0)
    )
    return jsonify({'addiction_risk': result})

@app.route('/predict/academic', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def predict_academic():
    logger.info(f"Solicitud recibida: {request.method} {request.path} from {request.remote_addr} con User-Agent: {request.headers.get('User-Agent')}")
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    result = academic_performance_risk(
        data.get('usage_hours', 0),
        data.get('sleep_hours', 0),
        data.get('mental_health_score', 0)
    )
    return jsonify({'academic_risk': result['risk'], 'probability': result['probability']})

@app.route('/predict/sleep', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def predict_sleep():
    logger.info(f"Solicitud recibida: {request.method} {request.path} from {request.remote_addr} con User-Agent: {request.headers.get('User-Agent')}")
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    result = sleep_prediction(
        data.get('usage_hours', 0),
        data.get('age', 0),
        data.get('mental_health_score', 0)
    )
    return jsonify({'sleep_hours': result})

@app.route('/stats/platforms', methods=['GET', 'OPTIONS'])
@limiter.limit("20 per minute")
def get_platform_stats():
    logger.info(f"Solicitud recibida: {request.method} {request.path} from {request.remote_addr} con User-Agent: {request.headers.get('User-Agent')}")
    if request.method == 'OPTIONS':
        return '', 200
    result = get_platform_distribution()
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)