from flask import Blueprint, jsonify, Response
from services.excel_service import load_dataset
import json

catalogs_bp = Blueprint('catalogs', __name__, url_prefix='/api/catalogs')


def build_label_value_response(values):
    seen = set()
    result = []

    for item in values:
        cleaned = item.strip()
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append({
                "value": cleaned,
                "label": cleaned.capitalize()
            })

    result_sorted = sorted(result, key=lambda x: x["label"].lower())
    return result_sorted


@catalogs_bp.route('/country', methods=['GET'])
def list_countries():
    try:
        df = load_dataset()
        countries = df['Country'].dropna().tolist()
        final_output = build_label_value_response(countries)
        return Response(json.dumps(final_output, ensure_ascii=False), content_type='application/json; charset=utf-8'), 200
    except Exception as e:
        print(f"Error en /api/catalogs/country: {str(e)}")
        return jsonify({"error": "Error interno al listar países"}), 500


@catalogs_bp.route('/academic_level', methods=['GET'])
def list_academic_levels():
    try:
        df = load_dataset()
        levels = df['Academic_Level'].dropna().tolist()
        final_output = build_label_value_response(levels)
        return jsonify(final_output), 200
    except Exception as e:
        print(f"Error en /api/catalogs/academic_level: {str(e)}")
        return jsonify({"error": "Error interno al listar niveles académicos"}), 500


@catalogs_bp.route('/gender', methods=['GET'])
def list_genders():
    try:
        df = load_dataset()
        genders = df['Gender'].dropna().tolist()
        final_output = build_label_value_response(genders)
        return jsonify(final_output), 200
    except Exception as e:
        print(f"Error en /api/catalogs/gender: {str(e)}")
        return jsonify({"error": "Error interno al listar géneros"}), 500


@catalogs_bp.route('/most_used_platform', methods=['GET'])
def list_most_used_platform():
    try:
        df = load_dataset()
        platforms = df['Most_Used_Platform'].dropna().tolist()
        final_output = build_label_value_response(platforms)
        return jsonify(final_output), 200
    except Exception as e:
        print(f"Error en /api/catalogs/most_used_platform: {str(e)}")
        return jsonify({"error": "Error interno al listar plataformas más utilizadas"}), 500


@catalogs_bp.route('/relationship_status', methods=['GET'])
def list_relationship_status():
    try:
        df = load_dataset()
        statuses = df['Relationship_Status'].dropna().tolist()
        final_output = build_label_value_response(statuses)
        return jsonify(final_output), 200
    except Exception as e:
        print(f"Error en /api/catalogs/relationship_status: {str(e)}")
        return jsonify({"error": "Error interno al listar estados de relación"}), 500
