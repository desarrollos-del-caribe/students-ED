from flask import Flask, request, jsonify, request
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
CORS(app)

@app.route('/api/validate-age', methods=['GET'])
def api_validate_age():
    response, status = validate_age()
    return response, status

@app.route('/api/validate-countries', methods=['GET'])
def api_validate_countries():
    response, status = validate_countries()
    return response, status

@app.route('/api/null-info', methods=['GET'])
def api_null_info():
    response, status = get_null_info()
    return response, status

@app.route('/api/statistics', methods=['GET'])
def api_statistics():
    response, status = get_statistics()
    return response, status

@app.route('/api/outliers', methods=['GET'])
def api_outliers():
    response, status = detect_outliers()
    return response, status

@app.route('/api/generate-plots', methods=['GET'])
def api_generate_plots():
    response, status = generate_plots()
    return response, status

@app.route('/api/linear-regression', methods=['GET'])
def api_linear_regression():
    response, status = linear_regression_analysis()
    return response, status

@app.route('/api/logistic-regression', methods=['GET'])
def api_logistic_regression():
    response, status = logistic_regression_analysis()
    return response, status

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
    app.run(debug=True)