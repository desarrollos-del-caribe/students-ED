from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar rutas nuevas
from routes.analysis import analysis_bp
from routes.catalogs import catalogs_bp

# Crear la aplicación Flask
app = Flask(__name__)

# Configurar CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://localhost:3000", "*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
        "supports_credentials": True
    }
})

# Configurar limitador de rate
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "200 per hour"],
    storage_uri="memory://"
)

# Registrar blueprints
app.register_blueprint(analysis_bp)
app.register_blueprint(catalogs_bp)

# Crear directorios necesarios al iniciar
def ensure_directories():
    try:
        os.makedirs('static/graphs')
    except FileExistsError:
        print("Los directorios ya existen, no es necesario crearlos.")

# Ruta principal de la API
@app.route('/api', methods=['GET'])
def api_home():
    """Endpoint principal de la API"""
    return jsonify({
        "message": "API de Análisis de Redes Sociales y Rendimiento Académico",
        "version": "1.0.0",
        "endpoints": {
            "users": "/api/users",
            "models": "/api/models", 
            "catalogs": "/api/catalogs",
            "analysis": "/api/analyze",
            "predictions": "/api/predictions",
            "visualizations": "/api/visualizations"
        },
        "status": "active",
        "documentation": "Consultar README.md para documentación completa"
    }), 200

@app.before_request
def handle_preflight():
    """Manejar requests OPTIONS para CORS"""
    if request.method == "OPTIONS":
        response = jsonify()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.errorhandler(404)
def not_found(error):
    """Manejar errores 404"""
    return jsonify({
        'error': 'Endpoint no encontrado',
        'message': 'La ruta solicitada no existe',
        'available_endpoints': [
            '/api',
            '/api/models',
            '/api/catalogs',
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Manejar errores 500"""
    return jsonify({
        'error': 'Error interno del servidor',
        'message': 'Ha ocurrido un error inesperado'
    }), 500

if __name__ == '__main__':
    print("🚀 Iniciando API de Análisis de Redes Sociales...")
    
    ensure_directories()
    app.run(host='0.0.0.0', port=5000, debug=True)
