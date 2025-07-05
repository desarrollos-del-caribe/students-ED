from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

# Importar rutas nuevas
from routes.users import users_bp
from routes.models import models_bp
from routes.analysis import analysis_bp
from routes.visualizations import visualizations_bp

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
app.register_blueprint(users_bp)
app.register_blueprint(models_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(visualizations_bp)

# Crear directorios necesarios al iniciar
def ensure_directories():
    """Crear directorios necesarios si no existen"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "excel")
    backup_dir = os.path.join(data_dir, "backups")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

# Inicializar archivos Excel al iniciar la aplicación
def initialize_excel_files():
    """Inicializar archivos Excel con datos de ejemplo"""
    try:
        from utils.excel_utils import UsersExcelHandler, ModelsExcelHandler, PredictionsExcelHandler
        
        # Inicializar handlers (esto creará archivos si no existen)
        users_handler = UsersExcelHandler()
        models_handler = ModelsExcelHandler()
        predictions_handler = PredictionsExcelHandler()
        
        print("✅ Archivos Excel inicializados correctamente")
        
    except Exception as e:
        print(f"⚠️ Error inicializando archivos Excel: {str(e)}")

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
            '/api/users',
            '/api/models',
            '/api/analyze',
            '/api/predictions',
            '/api/visualizations'
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
    
    # Crear directorios necesarios
    ensure_directories()
    print("📁 Directorios creados/verificados")
    
    # Inicializar archivos Excel
    initialize_excel_files()
    
    print("🌐 Servidor disponible en:")
    print("   - Local: http://localhost:5000")
    print("   - API: http://localhost:5000/api")
    print("📋 Endpoints principales:")
    print("   - GET /api/users - Gestión de usuarios")
    print("   - GET /api/models - Modelos ML disponibles")
    print("   - POST /api/analyze/user/{id} - Análisis completo")
    print("   - GET /api/predictions/{user_id} - Predicciones")
    print("   - GET /api/visualizations/dashboard - Dashboard completo")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
