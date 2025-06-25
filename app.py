from flask import Flask, request
from flask_cors import CORS
import os
import pandas as pd

# Servicios de análisis
from services import (
    validate_age,
    validate_countries,
    get_null_info,
    get_statistics,
    detect_outliers,
    generate_plots,
    linear_regression_analysis,
    logistic_regression_analysis,
    correlation_analysis,
    decision_tree_analysis
)


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'sql'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return {'error': 'No se encontró el archivo en la solicitud'}, 400

    file = request.files['file']

    if file.filename == '':
        return {'error': 'No se seleccionó ningún archivo'}, 400

    if not allowed_file(file.filename):
        return {'error': f'Formato no permitido. Solo se aceptan: {", ".join(ALLOWED_EXTENSIONS)}'}, 400

    try:
        extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"Students_Addiction.xlsx"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Leer y convertir a DataFrame según tipo
        if extension == 'csv':
            df = pd.read_csv(file)
        elif extension in ['xls', 'xlsx']:
            df = pd.read_excel(file)
        elif extension == 'json':
            df = pd.read_json(file)
        elif extension == 'sql':
            sql_content = file.read().decode('utf-8')
            return {'error': 'Archivos .sql no se pueden convertir directamente a Excel'}, 400
        else:
            return {'error': 'Formato no reconocido'}, 400

        # Guardar como Excel
        df.to_excel(filepath, index=False, engine='openpyxl')

        return {
            'message': f'Archivo convertido y guardado correctamente como {filename}',
            'filename': filename
        }, 200

    except Exception as e:
        return {'error': f'No se pudo guardar el archivo: {str(e)}'}, 500

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

@app.route('/api/correlation', methods=['GET'])
def api_correlation():
    response, status = correlation_analysis()
    return response, status

@app.route('/api/decision-tree', methods=['GET'])
def api_decision_tree():
    response, status = decision_tree_analysis()
    return response, status


if __name__ == '__main__':
    app.run(debug=True)
