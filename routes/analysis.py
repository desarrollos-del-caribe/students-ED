from flask import Blueprint, request, jsonify
from services.analysis_model import predict_mental_health_score, predict_sleep_hours, academic_performance_risk

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api/models')

@analysis_bp.route('/mental-health', methods=['POST'])
def predict_mental_health():
    """
    Endpoint para predecir el Mental_Health_Score de un usuario según sus hábitos.
    """
    try:
        data = request.get_json()

        # Validar campos requeridos
        required_fields = ["usage_hours", "sleep_hours", "addicted_score", "conflicts_score", "academic_impact"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Faltan campos requeridos en la solicitud."}), 400

        prediction = predict_mental_health_score(
            usage_hours=data["usage_hours"],
            sleep_hours=data["sleep_hours"],
            addicted_score=data["addicted_score"],
            conflicts_score=data["conflicts_score"],
            academic_impact=data["academic_impact"]
        )

        return jsonify(prediction), 200

    except Exception as e:
        print(f"Error en /predict/mental-health: {str(e)}")
        return jsonify({"error": f"Error al predecir salud mental: {str(e)}"}), 500

@analysis_bp.route('/sleep-prediction', methods=['POST'])
def predict_sleep():
    """
    Predice las horas de sueño estimadas según el uso diario de redes sociales.
    """
    try:
        data = request.get_json()
        if "usage_hours" not in data:
            return jsonify({"error": "El campo 'usage_hours' es obligatorio"}), 400

        result = predict_sleep_hours(data["usage_hours"])
        return jsonify(result), 200

    except Exception as e:
        print(f"Error en /sleep-prediction: {str(e)}")
        return jsonify({"error": "Error al predecir horas de sueño"}), 500

@analysis_bp.route('/academic-impact', methods=['POST'])
def predict_academic_impact_endpoint():
    """
    Endpoint para predecir si el uso de redes sociales afecta el rendimiento académico.
    """
    try:
        data = request.get_json()
        required = ["usage_hours", "sleep_hours", "mental_health_score"]
        if not all(k in data for k in required):
            return jsonify({"error": "Faltan campos requeridos"}), 400

        result = predict_academic_impact(
            usage_hours=data["usage_hours"],
            sleep_hours=data["sleep_hours"],
            mental_health_score=data["mental_health_score"]
        )
        return jsonify(result), 200

    except Exception as e:
        print(f"Error en /academic-impact: {str(e)}")
        return jsonify({"error": "Error al predecir impacto académico"}), 500

#Predecir el riesgo de que el rendimiento academico se vea afectado pos: horas de uso, salud mental y horas de sueño
@analysis_bp.route('/academic-risk', methods=['POST'])
def predict_academic_risk():
    """
    Predice el riesgo académico usando horas de uso, sueño y salud mental.
    """
    try:
        data = request.get_json()
        required = ["usage_hours", "sleep_hours", "mental_health_score"]
        if not all(k in data for k in required):
            return jsonify({"error": "Faltan campos requeridos"}), 400

        result = academic_performance_risk(
            usage_hours=data["usage_hours"],
            sleep_hours=data["sleep_hours"],
            mental_health_score=data["mental_health_score"]
        )
        return jsonify(result), 200

    except Exception as e:
        print(f"Error en /academic-risk: {str(e)}")
        return jsonify({"error": "Error al predecir riesgo académico"}), 500