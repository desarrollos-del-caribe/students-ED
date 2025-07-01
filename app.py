from flask import Flask, request, jsonify
from flask_cors import CORS
from services.analysis_model import (
    social_media_addiction_risk,
    academic_performance_risk,
    sleep_prediction
)
from config import Config

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "API funcionando correctamente"})


@app.route('/students/<history_model>', methods=['GET'])
def get_students(history_model):
    try:
        conn = Config.get_connection()
        cursor = conn.cursor(as_dict=True)

        query = """
            SELECT id, age, gender_id, academic_level_id, country_id, avg_daily_used_hours,
                   social_network_id, affects_academic_performance, sleep_hours_per_night,
                   mental_health_score, relationship_status_id, conflicts_over_social_media,
                   addicted_score
            FROM Tbl_Students_Model
            WHERE history_models_import_id = %s
        """
        cursor.execute(query, (history_model,))
        students = cursor.fetchall()
        conn.close()
        return jsonify({'students': students}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/addiction/<history_model>', methods=['GET', 'POST'])
def predict_addiction_route(history_model):
    if request.method == 'POST':
        data = request.get_json() or {}
        result = social_media_addiction_risk(
            usage_hours=data.get("usage_hours", 0),
            addicted_score=data.get("addicted_score", 0),
            mental_health_score=data.get("mental_health_score", 0),
            conflicts_score=data.get("conflicts_score", 0),
            historyModel=history_model
        )
        return jsonify({"prediction": result})

    usage_hours = float(request.args.get("usage_hours", 0))
    addicted_score = float(request.args.get("addicted_score", 0))
    mental_health_score = float(request.args.get("mental_health_score", 0))
    conflicts_score = float(request.args.get("conflicts_score", 0))

    result = social_media_addiction_risk(
        usage_hours=usage_hours,
        addicted_score=addicted_score,
        mental_health_score=mental_health_score,
        conflicts_score=conflicts_score,
        historyModel=history_model
    )
    return jsonify({"prediction": result})

@app.route('/predict/academic/<history_model>', methods=['GET', 'POST'])
def predict_academic_route(history_model):
    if request.method == 'POST':
        data = request.get_json() or {}
        result = academic_performance_risk(
            usage_hours=data.get("usage_hours", 0),
            sleep_hours=data.get("sleep_hours", 0),
            mental_health_score=data.get("mental_health_score", 0),
            historyModel=history_model
        )
        return jsonify(result)

    usage_hours = float(request.args.get("usage_hours", 0))
    sleep_hours = float(request.args.get("sleep_hours", 0))
    mental_health_score = float(request.args.get("mental_health_score", 0))

    result = academic_performance_risk(
        usage_hours=usage_hours,
        sleep_hours=sleep_hours,
        mental_health_score=mental_health_score,
        historyModel=history_model
    )
    return jsonify(result)

@app.route('/predict/sleep/<history_model>', methods=['GET', 'POST'])
def predict_sleep_route(history_model):
    if request.method == 'POST':
        data = request.get_json() or {}
        result = sleep_prediction(
            usage_hours=data.get("usage_hours", 0),
            age=data.get("age", 0),
            mental_health_score=data.get("mental_health_score", 0),
            historyModel=history_model
        )
        return jsonify({"predicted_sleep_hours": result})

    usage_hours = float(request.args.get("usage_hours", 0))
    age = int(request.args.get("age", 0))
    mental_health_score = float(request.args.get("mental_health_score", 0))

    result = sleep_prediction(
        usage_hours=usage_hours,
        age=age,
        mental_health_score=mental_health_score,
        historyModel=history_model
    )
    return jsonify({"predicted_sleep_hours": result})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=False)
