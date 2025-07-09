from flask import Blueprint, request, jsonify
from services.analysis_model import (
    predict_sleep_hours, academic_performance_risk, student_performance_prediction, 
    addiction_by_country, social_media_addiction_risk, visualize_decision_tree,
    run_kmeans_clustering, predict_academic_impact, plot_correlation_heatmap, analyze_user
)
from services.excel_service import load_dataset

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api/models')

@analysis_bp.route('/sleep-prediction', methods=['POST'])
def predict_sleep():
    """
    Predice las horas de sueño estimadas según el uso diario de redes sociales.
    """
    try:
        data = request.get_json()
        if "social_media_usage" not in data:
            return jsonify({"error": "El campo 'social_media_usage' es obligatorio"}), 400

        result = predict_sleep_hours(data)
        predicted_hours = result.get("predicted_sleep_hours", 0)
        recommendations = []
        if predicted_hours < 6:
            recommendations = [
                "Reduzca el tiempo de uso de redes sociales antes de dormir para mejorar la calidad del sueño.",
                "Establezca una rutina de sueño consistente, evitando pantallas al menos 1 hora antes de acostarse.",
                "Considere usar aplicaciones de control de tiempo en pantalla para limitar el uso de redes sociales."
            ]
        elif 6 <= predicted_hours < 8:
            recommendations = [
                "Mantenga un horario regular para acostarse y levantarse para optimizar el descanso.",
                "Practique técnicas de relajación como meditación antes de dormir para mejorar la calidad del sueño.",
                "Evite el uso prolongado de dispositivos electrónicos en la noche para no afectar el ritmo circadiano."
            ]
        else:
            recommendations = [
                "Continúe manteniendo un uso moderado de redes sociales para preservar un sueño saludable.",
                "Incorpore actividades relajantes como leer un libro antes de dormir para mantener la calidad del sueño.",
                "Monitoree regularmente sus hábitos de sueño para asegurar que se mantengan en un rango saludable."
            ]
        result["recommendations"] = recommendations
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
        
        required_fields = ["social_media_usage", "sleep_hours_per_night", "conflicts_over_social_media"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Faltan campos requeridos: social_media_usage, sleep_hours_per_night, conflicts_over_social_media"}), 400

        result = predict_academic_impact(data)
        impact = result.get("academic_impact", "Unknown")
        recommendations = []
        if impact.lower() == "high":
            recommendations = [
                "Limite el tiempo de uso de redes sociales durante las horas de estudio para mejorar la concentración.",
                "Busque apoyo académico, como tutorías, para mitigar el impacto en el rendimiento.",
                "Establezca metas de estudio claras y use técnicas como Pomodoro para mantener el enfoque."
            ]
        elif impact.lower() == "medium":
            recommendations = [
                "Programe descansos específicos para el uso de redes sociales y evite distracciones durante el estudio.",
                "Monitoree su rendimiento académico regularmente para identificar áreas de mejora.",
                "Participe en actividades extracurriculares que fomenten el equilibrio entre el estudio y el ocio."
            ]
        else:
            recommendations = [
                "Mantenga un equilibrio saludable entre el uso de redes sociales y las responsabilidades académicas.",
                "Continúe utilizando herramientas de organización como calendarios para gestionar el tiempo de estudio.",
                "Considere compartir sus estrategias de estudio con compañeros para fomentar buenas prácticas."
            ]
        result["recommendations"] = recommendations
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": "Error al predecir impacto académico"}), 500

@analysis_bp.route('/academic-risk', methods=['POST'])
def predict_academic_risk():
    """
    Predice el riesgo académico usando horas de uso, sueño y salud mental.
    """
    try:
        user_data = request.get_json()
        required = ["social_media_usage", "sleep_hours_per_night", "conflicts_over_social_media"]
        if not all(k in user_data for k in required):
            return jsonify({"error": "Faltan campos requeridos"}), 400

        result = academic_performance_risk(user_data)
        risk_level = result.get("risk_level", "Unknown")
        recommendations = []
        if risk_level.lower() == "high":
            recommendations = [
                "Reduzca significativamente el tiempo en redes sociales para minimizar el riesgo académico.",
                "Consulte con un consejero académico o psicólogo para abordar posibles problemas de salud mental.",
                "Establezca un horario de estudio estructurado para mejorar el rendimiento académico."
            ]
        elif risk_level.lower() == "medium":
            recommendations = [
                "Use aplicaciones de bloqueo de distracciones durante las horas de estudio para reducir el uso de redes sociales.",
                "Participe en grupos de estudio para mantenerse motivado y enfocado en las metas académicas.",
                "Priorice el sueño de calidad, asegurando al menos 7-8 horas por noche."
            ]
        else:
            recommendations = [
                "Mantenga los hábitos actuales de uso de redes sociales, ya que no representan un riesgo significativo.",
                "Continúe monitoreando su rendimiento académico para asegurar consistencia.",
                "Incorpore actividades de enriquecimiento académico, como cursos en línea o talleres."
            ]
        result["recommendations"] = recommendations
        return jsonify(result), 200

    except Exception as e:
        print(f"Error en /academic-risk: {str(e)}")
        return jsonify({"error": "Error al predecir riesgo académico"}), 500
    
@analysis_bp.route('/student-performance/<int:student_id>', methods=['GET'])
def get_student_performance(student_id):
    try:
        result = student_performance_prediction(student_id)
        performance = result.get("performance_level", "Unknown")
        recommendations = []
        if performance.lower() == "low":
            recommendations = [
                "Considere trabajar con un tutor para mejorar en áreas académicas específicas.",
                "Reduzca el tiempo en redes sociales y priorice el tiempo de estudio efectivo.",
                "Busque apoyo emocional o psicológico si el estrés está afectando el rendimiento."
            ]
        elif performance.lower() == "medium":
            recommendations = [
                "Establezca metas académicas claras y desarrolle un plan de estudio estructurado.",
                "Utilice recursos en línea, como videos educativos, para reforzar el aprendizaje.",
                "Mantenga un diálogo con profesores para recibir retroalimentación y mejorar."
            ]
        else:
            recommendations = [
                "Continúe con sus hábitos de estudio efectivos para mantener un alto rendimiento.",
                "Considere explorar temas avanzados o proyectos adicionales para seguir creciendo.",
                "Comparta sus estrategias de éxito con otros estudiantes para fomentar un entorno colaborativo."
            ]
        result["recommendations"] = recommendations
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Error al predecir rendimiento: {str(e)}"}), 500

@analysis_bp.route('/addiction-by-country', methods=['GET'])
def get_addiction_by_country():
    try:
        min_students = int(request.args.get('min_students', 5))
        result = addiction_by_country(min_students=min_students)
        recommendations = [
            "Promueva campañas de concienciación sobre el uso responsable de redes sociales en países con altos índices de adicción.",
            "Implemente programas educativos en escuelas para enseñar a los estudiantes sobre el impacto del uso excesivo de redes sociales.",
            "Fomente políticas públicas que regulen el acceso a redes sociales en horarios escolares."
        ]
        result["recommendations"] = recommendations
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Error en adicción por país: {str(e)}"}), 500 
    
@analysis_bp.route('/addiction-risk', methods=['POST'])
def predict_addiction_risk():
    """
    Endpoint para predecir el riesgo de adicción a redes sociales.
    """
    try:
        user_data = request.get_json()

        required_fields = ["social_media_usage", "sleep_hours_per_night", "conflicts_over_social_media"]
        if not all(field in user_data for field in required_fields):
            return jsonify({"error": "Faltan campos requeridos: social_media_usage, sleep_hours_per_night, conflicts_over_social_media."}), 400

        result = social_media_addiction_risk(user_data)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": "Error al predecir riesgo de adicción."}), 500
    
@analysis_bp.route('/tree-visualization', methods=['POST'])
def tree_visualization():
    """
    Endpoint para generar una visualización del árbol de decisión y recomendaciones personalizadas.
    """
    try:
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "No se proporcionaron datos del usuario."}), 400
        mapped_user_data = {
            "Avg_Daily_Usage_Hours": user_data.get("social_media_usage", 0),
            "Sleep_Hours_Per_Night": user_data.get("sleep_hours_per_night", 0),
            "Mental_Health_Score": user_data.get("mental_health_score", 5),  
            "Addicted_Score": user_data.get("addicted_score", 0),  
            "Conflicts_Over_Social_Media": user_data.get("conflicts_over_social_media", 0)
        }

        required_fields = [
            "Avg_Daily_Usage_Hours",
            "Sleep_Hours_Per_Night",
            "Mental_Health_Score",
            "Addicted_Score",
            "Conflicts_Over_Social_Media"
        ]
        if not all(field in mapped_user_data and mapped_user_data[field] is not None for field in required_fields):
            missing_fields = [field for field in required_fields if field not in mapped_user_data or mapped_user_data[field] is None]
            return jsonify({"error": f"Faltan campos requeridos: {', '.join(missing_fields)}."}), 400

        target = request.args.get("target", "Addicted_Score")

        data = visualize_decision_tree(target, mapped_user_data)
        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": f"Error al generar el análisis: {str(e)}"}), 500

@analysis_bp.route('/kmeans-clustering', methods=['POST'])
def kmeans_visualization():
    """
    Endpoint para analizar el comportamiento del usuario y agruparlo en un perfil.
    """
    try:
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "No se proporcionaron datos del usuario."}), 400

        required_fields = ["age", "social_media_usage", "sleep_hours_per_night", "conflicts_over_social_media"]
        if not all(field in user_data for field in required_fields):
            return jsonify({"error": f"Faltan campos requeridos: {', '.join(required_fields)}."}), 400

        n_clusters = int(request.args.get("clusters", 3))
        if n_clusters < 2 or n_clusters > 10:
            return jsonify({"error": "El número de grupos debe estar entre 2 y 10."}), 400

        data = run_kmeans_clustering(user_data, n_clusters)
        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": f"Error al analizar los datos: {str(e)}"}), 500

@analysis_bp.route('/correlation-heatmap', methods=['GET'])
def get_correlation_heatmap():
    try:
        df = load_dataset()
        graph_data = plot_correlation_heatmap(df)
        recommendations = [
            "Identifique las variables con alta correlación para enfocar intervenciones en esas áreas específicas.",
            "Use el mapa de calor para educar a los estudiantes sobre cómo el uso de redes sociales afecta otras variables.",
            "Realice un análisis más profundo de las correlaciones fuertes para diseñar estrategias preventivas."
        ]
        graph_data["recommendations"] = recommendations
        return jsonify({"graph_data": graph_data, "recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": f"Error generando el mapa de calor: {str(e)}"}), 500

@analysis_bp.route("/analyze-user", methods=["POST"])
def analyze_mental_route():
    try:
        user_data = request.get_json()
        result = analyze_user(user_data)
        mental_health_score = result.get("mental_health_score", 0)
        recommendations = []
        if mental_health_score < 30:
            recommendations = [
                "Busque apoyo profesional, como un terapeuta, para abordar problemas de salud mental.",
                "Reduzca el tiempo en redes sociales y priorice actividades que promuevan el bienestar emocional.",
                "Participe en actividades de relajación, como yoga o meditación, para mejorar la salud mental."
            ]
        elif 30 <= mental_health_score < 70:
            recommendations = [
                "Monitoree regularmente su salud mental y busque recursos como aplicaciones de bienestar.",
                "Establezca un equilibrio entre el uso de redes sociales y actividades sociales en persona.",
                "Haga ejercicios de atención plena para reducir el estrés y mejorar el enfoque."
            ]
        else:
            recommendations = [
                "Continúe manteniendo hábitos saludables para preservar una buena salud mental.",
                "Considere compartir estrategias de bienestar con otros para fomentar un entorno positivo.",
                "Mantenga un diario de gratitud para reforzar el bienestar emocional."
            ]
        result["recommendations"] = recommendations
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Error al analizar usuario: {str(e)}"}), 500