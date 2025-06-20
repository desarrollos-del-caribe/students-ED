from flask import Flask
from flask_cors import CORS
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
    decision_tree_analysis)

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