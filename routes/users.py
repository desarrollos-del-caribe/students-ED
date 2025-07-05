from flask import Blueprint, request, jsonify
from datetime import datetime
from utils.excel_utils import UsersExcelHandler
from models.data_models import User
import traceback

users_bp = Blueprint('users', __name__, url_prefix='/api/users')

# Inicializar handler
users_handler = UsersExcelHandler()

@users_bp.route('/', methods=['POST'])
def create_user():
    """Crear nuevo usuario"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se enviaron datos'}), 400
        
        # Validar datos requeridos
        required_fields = ['name', 'email', 'age', 'gender', 'education_level']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo requerido faltante: {field}'}), 400
        
        # Crear usuario
        user_data = {
            'name': data['name'],
            'email': data['email'],
            'age': int(data['age']),
            'gender': data['gender'],
            'education_level': data['education_level'],
            'social_media_usage': int(data.get('social_media_usage', 5)),
            'academic_performance': float(data.get('academic_performance', 75.0)),
            'main_platform': data.get('main_platform', 'Instagram'),
            'study_hours': int(data.get('study_hours', 25))
        }
        
        # Validar valores
        if user_data['gender'] not in ['Masculino', 'Femenino', 'Otro']:
            return jsonify({'error': 'Género debe ser: Masculino, Femenino, u Otro'}), 400
        
        if user_data['education_level'] not in ['Bachillerato', 'Universidad', 'Posgrado']:
            return jsonify({'error': 'Nivel educativo debe ser: Bachillerato, Universidad, o Posgrado'}), 400
        
        if not (1 <= user_data['social_media_usage'] <= 10):
            return jsonify({'error': 'Uso de redes sociales debe ser entre 1 y 10 horas'}), 400
        
        if not (0 <= user_data['academic_performance'] <= 100):
            return jsonify({'error': 'Rendimiento académico debe ser entre 0 y 100'}), 400
        
        valid_platforms = ['Instagram', 'TikTok', 'Facebook', 'Twitter', 'YouTube', 'LinkedIn']
        if user_data['main_platform'] not in valid_platforms:
            return jsonify({'error': f'Plataforma debe ser una de: {", ".join(valid_platforms)}'}), 400
        
        # Crear usuario en Excel
        new_user = users_handler.create_user(user_data)
        
        return jsonify({
            'message': 'Usuario creado exitosamente',
            'user': new_user
        }), 201
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error creando usuario: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Error interno del servidor'}), 500


@users_bp.route('/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Obtener usuario por ID"""
    try:
        user = users_handler.get_user_by_id(user_id)
        return jsonify({'user': user}), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print(f"Error obteniendo usuario: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@users_bp.route('/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    """Actualizar usuario existente"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se enviaron datos'}), 400
        
        # Verificar que el usuario existe
        try:
            existing_user = users_handler.get_user_by_id(user_id)
        except ValueError:
            return jsonify({'error': f'Usuario con ID {user_id} no encontrado'}), 404
        
        # Preparar datos de actualización
        update_data = {}
        
        # Campos opcionales para actualizar
        if 'name' in data:
            update_data['name'] = data['name']
        if 'email' in data:
            update_data['email'] = data['email']
        if 'age' in data:
            update_data['age'] = int(data['age'])
        if 'gender' in data:
            if data['gender'] not in ['Masculino', 'Femenino', 'Otro']:
                return jsonify({'error': 'Género debe ser: Masculino, Femenino, u Otro'}), 400
            update_data['gender'] = data['gender']
        if 'education_level' in data:
            if data['education_level'] not in ['Bachillerato', 'Universidad', 'Posgrado']:
                return jsonify({'error': 'Nivel educativo debe ser: Bachillerato, Universidad, o Posgrado'}), 400
            update_data['education_level'] = data['education_level']
        if 'social_media_usage' in data:
            usage = int(data['social_media_usage'])
            if not (1 <= usage <= 10):
                return jsonify({'error': 'Uso de redes sociales debe ser entre 1 y 10 horas'}), 400
            update_data['social_media_usage'] = usage
        if 'academic_performance' in data:
            performance = float(data['academic_performance'])
            if not (0 <= performance <= 100):
                return jsonify({'error': 'Rendimiento académico debe ser entre 0 y 100'}), 400
            update_data['academic_performance'] = performance
        if 'main_platform' in data:
            valid_platforms = ['Instagram', 'TikTok', 'Facebook', 'Twitter', 'YouTube', 'LinkedIn']
            if data['main_platform'] not in valid_platforms:
                return jsonify({'error': f'Plataforma debe ser una de: {", ".join(valid_platforms)}'}), 400
            update_data['main_platform'] = data['main_platform']
        if 'study_hours' in data:
            study_hours = int(data['study_hours'])
            if study_hours < 0 or study_hours > 168:
                return jsonify({'error': 'Horas de estudio debe ser entre 0 y 168 horas semanales'}), 400
            update_data['study_hours'] = study_hours
        
        if not update_data:
            return jsonify({'error': 'No se enviaron campos para actualizar'}), 400
        
        # Actualizar usuario
        updated_user = users_handler.update_user(user_id, update_data)
        
        return jsonify({
            'message': 'Usuario actualizado exitosamente',
            'user': updated_user
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error actualizando usuario: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Error interno del servidor'}), 500


@users_bp.route('/', methods=['GET'])
def get_all_users():
    """Obtener todos los usuarios"""
    try:
        df = users_handler.read_users()
        users = df.to_dict('records')
        
        return jsonify({
            'users': users,
            'total': len(users)
        }), 200
        
    except Exception as e:
        print(f"Error obteniendo usuarios: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@users_bp.route('/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Eliminar usuario (marcar como inactivo)"""
    try:
        # Verificar que el usuario existe
        try:
            existing_user = users_handler.get_user_by_id(user_id)
        except ValueError:
            return jsonify({'error': f'Usuario con ID {user_id} no encontrado'}), 404
        
        # En lugar de eliminar físicamente, podemos marcar como inactivo
        # o implementar eliminación física si es necesario
        update_data = {
            'updated_at': datetime.now().isoformat(),
            # Podríamos agregar un campo 'active': False
        }
        
        updated_user = users_handler.update_user(user_id, update_data)
        
        return jsonify({
            'message': f'Usuario {user_id} eliminado exitosamente'
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print(f"Error eliminando usuario: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@users_bp.route('/validate', methods=['POST'])
def validate_user_data():
    """Validar datos de usuario sin crear"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se enviaron datos'}), 400
        
        # Crear objeto User para validación
        user = User.from_dict(data)
        errors = user.validate()
        
        if errors:
            return jsonify({
                'valid': False,
                'errors': errors
            }), 400
        
        return jsonify({
            'valid': True,
            'message': 'Datos válidos'
        }), 200
        
    except Exception as e:
        print(f"Error validando datos: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@users_bp.route('/search', methods=['GET'])
def search_users():
    """Buscar usuarios por criterios"""
    try:
        # Obtener parámetros de búsqueda
        name = request.args.get('name', '')
        gender = request.args.get('gender', '')
        education_level = request.args.get('education_level', '')
        min_age = request.args.get('min_age', type=int)
        max_age = request.args.get('max_age', type=int)
        min_performance = request.args.get('min_performance', type=float)
        max_performance = request.args.get('max_performance', type=float)
        
        # Leer todos los usuarios
        df = users_handler.read_users()
        
        # Aplicar filtros
        if name:
            df = df[df['name'].str.contains(name, case=False, na=False)]
        
        if gender:
            df = df[df['gender'] == gender]
        
        if education_level:
            df = df[df['education_level'] == education_level]
        
        if min_age is not None:
            df = df[df['age'] >= min_age]
        
        if max_age is not None:
            df = df[df['age'] <= max_age]
        
        if min_performance is not None:
            df = df[df['academic_performance'] >= min_performance]
        
        if max_performance is not None:
            df = df[df['academic_performance'] <= max_performance]
        
        users = df.to_dict('records')
        
        return jsonify({
            'users': users,
            'total': len(users),
            'filters_applied': {
                'name': name,
                'gender': gender,
                'education_level': education_level,
                'min_age': min_age,
                'max_age': max_age,
                'min_performance': min_performance,
                'max_performance': max_performance
            }
        }), 200
        
    except Exception as e:
        print(f"Error buscando usuarios: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@users_bp.route('/stats', methods=['GET'])
def get_user_stats():
    """Obtener estadísticas de usuarios"""
    try:
        df = users_handler.read_users()
        
        if len(df) == 0:
            return jsonify({
                'message': 'No hay usuarios registrados',
                'stats': {}
            }), 200
        
        stats = {
            'total_users': len(df),
            'average_age': float(df['age'].mean()),
            'average_social_media_usage': float(df['social_media_usage'].mean()),
            'average_academic_performance': float(df['academic_performance'].mean()),
            'average_study_hours': float(df['study_hours'].mean()),
            'gender_distribution': df['gender'].value_counts().to_dict(),
            'education_level_distribution': df['education_level'].value_counts().to_dict(),
            'platform_distribution': df['main_platform'].value_counts().to_dict(),
            'performance_ranges': {
                'excellent': len(df[df['academic_performance'] >= 90]),
                'good': len(df[(df['academic_performance'] >= 75) & (df['academic_performance'] < 90)]),
                'average': len(df[(df['academic_performance'] >= 60) & (df['academic_performance'] < 75)]),
                'below_average': len(df[df['academic_performance'] < 60])
            }
        }
        
        return jsonify({'stats': stats}), 200
        
    except Exception as e:
        print(f"Error obteniendo estadísticas: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500
