import pandas as pd
import pymssql
import json
import logging
import re
from config import Config
from io import StringIO

logger = logging.getLogger(__name__)

COLUMN_SYNONYMS = {
    'id': ['studentid', 'student_id', 'id', 'sid'],
    'age': ['age', 'edad', 'years'],
    'gender_id': ['gender', 'sex', 'gender_id'],
    'academic_level_id': ['academiclevel', 'academic_level', 'education_level'],
    'country_id': ['country', 'pais', 'nation', 'country_id'],
    'avg_daily_used_hours': ['avgdailyusagehours', 'avg_daily_usage_hours', 'daily_usage', 'hours_used'],
    'social_network_id': ['mostusedplatform', 'most_used_platform', 'platform', 'social_media', 'platform_id'],
    'affects_academic_performance': ['affectsacademicperformance', 'affects_academic', 'academic_impact'],
    'sleep_hours_per_night': ['sleephourspernight', 'sleep_hours', 'sleep'],
    'mental_health_score': ['mentalhealthscore', 'mental_health', 'mental_score'],
    'relationship_status_id': ['relationshipstatus', 'relationship', 'status'],
    'conflicts_over_social_media': ['conflictsoversocialmedia', 'conflicts', 'social_media_conflicts'],
    'addicted_score': ['addictedscore', 'addiction_score', 'addicted'],
    'history_models_import_id': ['historyid', 'import_id', 'history_models_import_id', 'student_id']
}

# Prefijos de columnas dummy
DUMMY_PREFIXES = {
    'gender_id': ['gender_female', 'gender_male'],
    'academic_level_id': ['academiclevel_graduate', 'academiclevel_high school', 'academiclevel_undergraduate'],
    'relationship_status_id': ['relationshipstatus_single', 'relationshipstatus_in relationship', 'relationshipstatus_complicated'],
    'social_network_id': [
        'mostusedplatform_facebook', 'mostusedplatform_instagram', 'mostusedplatform_twitter',
        'mostusedplatform_tiktok', 'mostusedplatform_whatsapp', 'mostusedplatform_youtube',
        'mostusedplatform_linkedin', 'mostusedplatform_snapchat', 'mostusedplatform_wechat',
        'mostusedplatform_vkontakte', 'mostusedplatform_kakaotalk', 'mostusedplatform_line'
    ],
    'country_id': ['country_mexico', 'country_usa', 'country_brazil'],
    'affects_academic_performance': ['affects_yes', 'affects_no']
}

# Mapeo estático como respaldo
STATIC_MAPPINGS = {
    'gender_id': {
        'male': 1, 'female': 2, 'masculino': 1, 'femenino': 2,
        'true': 1, 'false': 2, 'verdadero': 1, 'falso': 2,
        'gender_male': 1, 'gender_female': 2
    },
    'academic_level_id': {
        'high school': 1, 'undergraduate': 2, 'graduate': 3,
        'high_school': 1, 'under_grad': 2, 'grad': 3,
        'academiclevel_high school': 1, 'academiclevel_undergraduate': 2, 'academiclevel_graduate': 3
    },
    'country_id': {
        'usa': 1, 'mexico': 2, 'méxico': 2, 'brazil': 3
    },
    'social_network_id': {
        'facebook': 1, 'instagram': 2, 'twitter': 3, 'tiktok': 4, 'whatsapp': 5,
        'youtube': 6, 'linkedin': 7, 'snapchat': 8, 'wechat': 9, 'vkontakte': 10,
        'kakaotalk': 11, 'line': 12,
        'mostusedplatform_facebook': 1, 'mostusedplatform_instagram': 2, 'mostusedplatform_twitter': 3,
        'mostusedplatform_tiktok': 4, 'mostusedplatform_whatsapp': 5, 'mostusedplatform_youtube': 6,
        'mostusedplatform_linkedin': 7, 'mostusedplatform_snapchat': 8, 'mostusedplatform_wechat': 9,
        'mostusedplatform_vkontakte': 10, 'mostusedplatform_kakaotalk': 11, 'mostusedplatform_line': 12
    },
    'relationship_status_id': {
        'single': 1, 'in relationship': 2, 'complicated': 3,
        'soltero': 1, 'en relacion': 2, 'complicado': 3,
        'relationshipstatus_single': 1, 'relationshipstatus_in relationship': 2, 'relationshipstatus_complicated': 3
    },
    'affects_academic_performance': {
        'yes': 1, 'no': 0, 'si': 1, 'no': 0,
        'true': 1, 'false': 0, 'verdadero': 1, 'falso': 0
    }
}

def get_categorical_mapping(conn, table_name, possible_columns=['gender', 'gender_name', 'name']):
    """Obtiene mapeos categóricos desde una tabla de la base de datos."""
    try:
        cursor = conn.cursor(as_dict=True)
        for col in possible_columns:
            try:
                cursor.execute(f"SELECT id, {col} FROM {table_name}")
                mapping = {row[col].lower(): row['id'] for row in cursor.fetchall() if row[col] is not None}
                if mapping:
                    cursor.close()
                    return mapping
            except Exception:
                continue
        cursor.close()
        logger.error(f"No se encontró columna válida en {table_name}, usando mapeo estático")
        return STATIC_MAPPINGS.get(table_name.lower(), {})
    except Exception as e:
        logger.error(f"Error obteniendo mapeo para {table_name}: {str(e)}")
        return STATIC_MAPPINGS.get(table_name.lower(), {})

def extract_numeric(value):
    """Extrae la parte numérica de una cadena."""
    if pd.isna(value):
        return 0
    try:
        match = re.search(r'\d+$', str(value))
        return int(match.group()) if match else 0
    except (ValueError, TypeError):
        return 0

def normalize_column_names(df):
    """Normaliza nombres de columnas y consolida columnas dummy."""
    normalized_columns = {}
    dummy_values = {key: [] for key in DUMMY_PREFIXES.keys()}

    for col in df.columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        matched = False

        # Buscar en sinónimos
        for standard_col, synonyms in COLUMN_SYNONYMS.items():
            if any(s.lower().replace(' ', '_').replace('-', '_') == col_lower for s in synonyms):
                normalized_columns[col] = standard_col
                matched = True
                break

        # Identificar columnas dummy
        if not matched:
            for standard_col, prefixes in DUMMY_PREFIXES.items():
                for prefix in prefixes:
                    if col_lower.startswith(prefix.lower().replace(' ', '_').replace('-', '_')):
                        dummy_values[standard_col].append(col)
                        matched = True
                        break

        if not matched:
            normalized_columns[col] = f"drop_{col}"  # Marcar para eliminación

    # Renombrar columnas
    df = df.rename(columns=normalized_columns)

    # Consolidar columnas dummy
    for standard_col, dummy_cols in dummy_values.items():
        if dummy_cols:
            df[standard_col] = 0
            for dummy_col in dummy_cols:
                if dummy_col in df.columns:
                    df[dummy_col] = df[dummy_col].astype(str).str.lower().map({
                        'verdadero': 1, 'falso': 0, 'true': 1, 'false': 0, '1': 1, '0': 0
                    }).fillna(0).astype(int)
                    df.loc[df[dummy_col] == 1, standard_col] = STATIC_MAPPINGS[standard_col].get(
                        dummy_col.lower().replace(' ', '_').replace('-', '_').split('_')[-1], 0
                    )
            df = df.drop(columns=[col for col in dummy_cols if col in df.columns])

    # Eliminar columnas marcadas con "drop_"
    df = df[[col for col in df.columns if not col.startswith('drop_')]]
    
    # Asegurar que todas las columnas requeridas estén presentes
    required_columns = [
        'age', 'gender_id', 'academic_level_id', 'country_id', 'avg_daily_used_hours',
        'social_network_id', 'affects_academic_performance', 'sleep_hours_per_night',
        'mental_health_score', 'relationship_status_id', 'conflicts_over_social_media',
        'addicted_score', 'history_models_import_id'
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0 if col not in ['history_models_import_id', 'mental_health_score', 'sleep_hours_per_night'] else None

    # Extraer history_models_import_id de id si está presente
    if 'id' in df.columns:
        df['history_models_import_id'] = df['id'].apply(extract_numeric)

    return df

def map_to_ids(df, column_name, conn):
    """Mapea valores categóricos a IDs consultando la base de datos o usando mapeo estático."""
    table_mapping = {
        'gender_id': 'Tbl_Genders',
        'academic_level_id': 'Tbl_AcademicLevels',
        'country_id': 'Tbl_Countries',
        'social_network_id': 'Tbl_SocialNetworks',
        'relationship_status_id': 'Tbl_RelationshipStatuses',
        'affects_academic_performance': 'Tbl_AcademicImpacts'
    }
    table_name = table_mapping.get(column_name)
    possible_columns = ['gender', 'gender_name', 'name'] if column_name == 'gender_id' else ['name', 'level', 'status']
    mapping = get_categorical_mapping(conn, table_name, possible_columns) if table_name else STATIC_MAPPINGS.get(column_name, {})

    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).str.lower().map(mapping).fillna(0).astype(int)
    else:
        df[column_name] = 0
    return df[column_name]

def process_sql_file(file):
    """Procesa un archivo SQL con comandos INSERT."""
    try:
        sql_content = file.read().decode('utf-8')
        data = []
        for line in sql_content.split(';'):
            line = line.strip()
            if line.upper().startswith('INSERT INTO'):
                values_start = line.find('VALUES') + 6
                values_str = line[values_start:].strip('() ')
                values = [v.strip("' ") for v in values_str.split(',')]
                if len(values) >= 13:
                    data.append({
                        'id': values[0] if values[0].isdigit() else None,
                        'age': int(values[1]) if values[1].replace('.', '').isdigit() else 18,
                        'gender_id': int(values[2]) if values[2].isdigit() else 0,
                        'academic_level_id': int(values[3]) if values[3].isdigit() else 0,
                        'country_id': int(values[4]) if values[4].isdigit() else 0,
                        'avg_daily_used_hours': float(values[5]) if values[5].replace('.', '').replace('-', '').replace('+', '').isdigit() else 0.0,
                        'social_network_id': int(values[6]) if values[6].isdigit() else 0,
                        'affects_academic_performance': int(values[7]) if values[7].isdigit() else 0,
                        'sleep_hours_per_night': float(values[8]) if values[8].replace('.', '').replace('-', '').replace('+', '').isdigit() else 6.0,
                        'mental_health_score': int(values[9]) if values[9].isdigit() else None,
                        'relationship_status_id': int(values[10]) if values[10].isdigit() else 0,
                        'conflicts_over_social_media': int(values[11]) if values[11].isdigit() else 0,
                        'addicted_score': int(values[12]) if values[12].isdigit() else 0,
                        'history_models_import_id': extract_numeric(values[13]) if len(values) > 13 else 0
                    })
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error procesando archivo SQL: {str(e)}")
        return None

def process_and_insert_data(file):
    try:
        # Leer archivo según su extensión
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        elif file_extension == 'json':
            data = json.load(file)
            df = pd.DataFrame(data)
        elif file_extension == 'sql':
            df = process_sql_file(file)
            if df is None:
                return {'success': False, 'error': 'Error procesando archivo SQL'}
        else:
            return {'success': False, 'error': f'Formato de archivo no soportado: {file_extension}'}

        logger.info(f"Columnas originales en el archivo: {df.columns.tolist()}")

        # Normalizar nombres de columnas
        df = normalize_column_names(df)
        logger.info(f"Columnas normalizadas: {df.columns.tolist()}")

        # Conectar a la base de datos
        conn = Config.get_connection()
        cursor = conn.cursor()

        # Mapear valores categóricos a IDs
        categorical_columns = ['gender_id', 'academic_level_id', 'country_id', 'social_network_id', 'relationship_status_id', 'affects_academic_performance']
        for col in categorical_columns:
            df[col] = map_to_ids(df, col, conn)

        # Asegurar que todas las columnas requeridas estén presentes
        required_columns = [
            'age', 'gender_id', 'academic_level_id', 'country_id', 'avg_daily_used_hours',
            'social_network_id', 'affects_academic_performance', 'sleep_hours_per_night',
            'mental_health_score', 'relationship_status_id', 'conflicts_over_social_media',
            'addicted_score', 'history_models_import_id'
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0 if col not in ['history_models_import_id', 'mental_health_score', 'sleep_hours_per_night'] else None

        # Validar y convertir tipos de datos
        type_conversions = {
            'age': (int, 18),
            'gender_id': (int, 0),
            'academic_level_id': (int, 0),
            'country_id': (int, 0),
            'avg_daily_used_hours': (float, 0.0),
            'social_network_id': (int, 0),
            'affects_academic_performance': (int, 0),
            'sleep_hours_per_night': (float, 6.0),
            'mental_health_score': (int, 50),
            'relationship_status_id': (int, 0),
            'conflicts_over_social_media': (int, 0),
            'addicted_score': (int, 0),
            'history_models_import_id': (int, 0)
        }

        for col, (dtype, default) in type_conversions.items():
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default).astype(dtype)
                except Exception as e:
                    logger.error(f"Error convirtiendo columna {col}: {str(e)}")
                    df[col] = default

        # Validar id por separado
        if 'id' in df.columns:
            df['id'] = df['id'].apply(lambda x: int(x) if pd.notna(x) and str(x).isdigit() else None)

        # Insertar datos en Tbl_Students_Model
        try:
            for _, row in df.iterrows():
                try:
                    include_id = 'id' in df.columns and pd.notnull(row['id']) and row['id'] is not None
                    columns = [
                        'age', 'gender_id', 'academic_level_id', 'country_id', 'avg_daily_used_hours',
                        'social_network_id', 'affects_academic_performance', 'sleep_hours_per_night',
                        'mental_health_score', 'relationship_status_id', 'conflicts_over_social_media',
                        'addicted_score', 'history_models_import_id'
                    ]
                    values = (
                        int(row['age']) if pd.notnull(row['age']) else 18,
                        int(row['gender_id']) if pd.notnull(row['gender_id']) else 0,
                        int(row['academic_level_id']) if pd.notnull(row['academic_level_id']) else 0,
                        int(row['country_id']) if pd.notnull(row['country_id']) else 0,
                        float(row['avg_daily_used_hours']) if pd.notnull(row['avg_daily_used_hours']) else 0.0,
                        int(row['social_network_id']) if pd.notnull(row['social_network_id']) else 0,
                        int(row['affects_academic_performance']) if pd.notnull(row['affects_academic_performance']) else 0,
                        float(row['sleep_hours_per_night']) if pd.notnull(row['sleep_hours_per_night']) else 6.0,
                        int(row['mental_health_score']) if pd.notnull(row['mental_health_score']) else 50,
                        int(row['relationship_status_id']) if pd.notnull(row['relationship_status_id']) else 0,
                        int(row['conflicts_over_social_media']) if pd.notnull(row['conflicts_over_social_media']) else 0,
                        int(row['addicted_score']) if pd.notnull(row['addicted_score']) else 0,
                        int(row['history_models_import_id']) if pd.notnull(row['history_models_import_id']) else 0
                    )

                    if include_id:
                        cursor.execute("SET IDENTITY_INSERT Tbl_Students_Model ON")
                        columns = ['id'] + columns
                        values = (int(row['id']),) + values
                        insert_query = f"""
                            INSERT INTO Tbl_Students_Model ({', '.join(columns)})
                            VALUES ({', '.join(['%s'] * len(columns))})
                        """
                    else:
                        insert_query = f"""
                            INSERT INTO Tbl_Students_Model ({', '.join(columns)})
                            VALUES ({', '.join(['%s'] * len(columns))})
                        """

                    cursor.execute(insert_query, values)
                except Exception as e:
                    logger.error(f"Error insertando fila: {str(e)}")
                    if include_id:
                        cursor.execute("SET IDENTITY_INSERT Tbl_Students_Model OFF")
                    conn.rollback()
                    conn.close()
                    return {'success': False, 'error': f"Error insertando fila: {str(e)}"}

            conn.commit()
        except Exception as e:
            logger.error(f"Error en inserción: {str(e)}")
            if 'include_id' in locals() and include_id:
                cursor.execute("SET IDENTITY_INSERT Tbl_Students_Model OFF")
            conn.rollback()
            conn.close()
            return {'success': False, 'error': f"Error en inserción: {str(e)}"}
        
        conn.close()
        return {'success': True, 'message': 'Datos insertados exitosamente'}
    except Exception as e:
        logger.error(f"Error procesando archivo: {str(e)}")
        if 'conn' in locals() and conn:
            if 'cursor' in locals() and cursor and 'include_id' in locals() and include_id:
                cursor.execute("SET IDENTITY_INSERT Tbl_Students_Model OFF")
            conn.close()
        return {'success': False, 'error': f"Error procesando archivo: {str(e)}"}