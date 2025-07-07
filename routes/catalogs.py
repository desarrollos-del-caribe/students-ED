from flask import Blueprint, jsonify,Response
from services.excel_service import load_dataset
import json

catalogs_bp = Blueprint('catalogs', __name__, url_prefix='/api/catalogs')

@catalogs_bp.route('/country', methods=['GET'])
def list_countries():
    """
    Endpoint para listar los países ordenados alfabéticamente,
    sin duplicados y con soporte UTF-8.
    """
    try:
        df = load_dataset()
        countries = df['Country'].dropna().tolist()

        seen = set()
        result = []
        for country in countries:
            cleaned = country.strip()
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                result.append(cleaned)

        result_sorted = sorted(result, key=lambda x: x.lower())
        final_output = [{"name": name} for name in result_sorted]

        return Response(json.dumps(final_output, ensure_ascii=False), content_type='application/json; charset=utf-8'), 200
        # return jsonify(final_output), 200
    except Exception as e:
        print(f"Error en /api/catalogs/country: {str(e)}")
        return jsonify({"error": "Error interno al listar países"}), 500

@catalogs_bp.route('/academic_level', methods=['GET'])
def list_academic_levels():
    """
    Endpoint para listar los niveles académicos ordenados alfabéticamente,
    sin duplicados y con soporte UTF-8.
    """
    try:
        df = load_dataset()
        academic_levels = df['Academic_Level'].dropna().tolist()

        seen = set()
        result = []
        for level in academic_levels:
            cleaned = level.strip()
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                result.append(cleaned)

        result_sorted = sorted(result, key=lambda x: x.lower())
        final_output = [{"name": name} for name in result_sorted]
        return jsonify(final_output), 200
    except Exception as e:
        print(f"Error en /api/catalogs/academic_level: {str(e)}")
        return jsonify({"error": "Error interno al listar niveles académicos"}), 500
    
@catalogs_bp.route('/gender', methods=['GET'])
def list_genders():
    """
    Endpoint para listar los géneros ordenados alfabéticamente,
    sin duplicados y con soporte UTF-8.
    """
    try:
        df = load_dataset()
        genders = df['Gender'].dropna().tolist()

        seen = set()
        result = []
        for gender in genders:
            cleaned = gender.strip()
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                result.append(cleaned)

        result_sorted = sorted(result, key=lambda x: x.lower())
        final_output = [{"name": name} for name in result_sorted]
        return jsonify(final_output), 200
    except Exception as e:
        print(f"Error en /api/catalogs/gender: {str(e)}")
        return jsonify({"error": "Error interno al listar géneros"}), 500


@catalogs_bp.route('/most_used_platform', methods=['GET'])
def list_most_used_platform():
    """
    Endpoint para listar las plataformas más utilizadas ordenadas alfabéticamente,
    sin duplicados y con soporte UTF-8.
    """
    try:
        df = load_dataset()
        platforms = df['Most_Used_Platform'].dropna().tolist()

        seen = set()
        result = []
        for platform in platforms:
            cleaned = platform.strip()
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                result.append(cleaned)

        result_sorted = sorted(result, key=lambda x: x.lower())
        final_output = [{"name": name} for name in result_sorted]
        return jsonify(final_output), 200
    except Exception as e:
        print(f"Error en /api/catalogs/most_used_platform: {str(e)}")
        return jsonify({"error": "Error interno al listar plataformas más utilizadas"}), 500
    
@catalogs_bp.route('/relationship_status', methods=['GET'])
def list_relationship_status():
    """
    Endpoint para listar los estados de relación ordenados alfabéticamente,
    sin duplicados y con soporte UTF-8.
    """
    try:
        df = load_dataset()
        statuses = df['Relationship_Status'].dropna().tolist()

        seen = set()
        result = []
        for status in statuses:
            cleaned = status.strip()
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                result.append(cleaned)

        result_sorted = sorted(result, key=lambda x: x.lower())
        final_output = [{"name": name} for name in result_sorted]
        return jsonify(final_output), 200
    except Exception as e:
        print(f"Error en /api/catalogs/relationship_status: {str(e)}")
        return jsonify({"error": "Error interno al listar estados de relación"}), 500