import pandas as pd
from config import Config
import pymssql
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_to_ids(df):
    try:
        conn = Config.get_connection()
        cursor = conn.cursor()
        
        # Mapear géneros
        cursor.execute("SELECT id, name_gender FROM Tbl_Genders")
        gender_map = {row[1].lower(): row[0] for row in cursor.fetchall()}
        df['gender_id'] = df.apply(
            lambda x: gender_map.get(str(x.get('gender', '')).lower(), 
                next((k for k, v in gender_map.items() if v == 1 and x.get('gendermale', False) or v == 2 and x.get('genderfemale', False)), 3)), 
            axis=1
        )
        
        # Mapear países
        cursor.execute("SELECT id, name_country FROM Tbl_Countries")
        country_map = {row[1].lower(): row[0] for row in cursor.fetchall()}
        df['country_id'] = df.apply(
            lambda x: country_map.get(str(x.get('country', '')).lower(), 1), 
            axis=1
        )
        
        # Mapear niveles académicos
        cursor.execute("SELECT id, name_level FROM Tbl_Academic_Levels")
        academic_map = {row[1].lower(): row[0] for row in cursor.fetchall()}
        df['academic_level_id'] = df.apply(
            lambda x: academic_map.get(str(x.get('academiclevel', '')).lower(), 
                next((
                    k for k, v in academic_map.items() 
                    if ('undergraduate' in v.lower() and x.get('academiclevelundergraduate', False)) or
                    ('graduate' in v.lower() and x.get('academiclevelgraduate', False)) or
                    ('high school' in v.lower() and x.get('academiclevelhigh school', False))
                ), 1)), 
            axis=1
        )
        
        # Mapear redes sociales
        cursor.execute("SELECT id, name_social_network FROM Tbl_Socials_Networks")
        social_map = {row[1].lower(): row[0] for row in cursor.fetchall()}
        df['social_network_id'] = df.apply(
            lambda x: social_map.get(str(x.get('mostusedplatform', '')).lower(), 
                next((
                    k for k, v in social_map.items() 
                    if any(col.startswith('mostusedplatform') and x.get(col, False) and v.lower() in col.replace('mostusedplatform', '').lower() 
                           for col in df.columns)
                ), 1)), 
            axis=1
        )
        
        # Mapear estado de relación
        cursor.execute("SELECT id, name_status FROM Tbl_RelationsShips_Status")
        relation_map = {row[1].lower(): row[0] for row in cursor.fetchall()}
        df['relationship_status_id'] = df.apply(
            lambda x: relation_map.get(str(x.get('relationshipstatus', '')).lower(), 
                next((
                    k for k, v in relation_map.items() 
                    if any(col.startswith('relationshipstatus') and x.get(col, False) and v.lower() in col.replace('relationshipstatus', '').lower() 
                           for col in df.columns)
                ), 1)), 
            axis=1
        )
        
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error en map_to_ids: {str(e)}")
        if 'conn' in locals():
            conn.close()
        raise e

def process_and_insert_data(file):
    try:
        content = BytesIO(file.read())
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(content)
        elif file_extension == 'xlsx':
            df = pd.read_excel(content)
        else:
            return {'error': 'Formato de archivo no soportado'}
        
        df.columns = [col.lower().replace('_', '') for col in df.columns]
        logger.info(f"Columnas encontradas en el archivo: {df.columns.tolist()}")
        
        df = map_to_ids(df)
        
        conn = Config.get_connection()
        cursor = conn.cursor()
        file_name = file.filename
        
        # Verificar si el archivo ya fue procesado
        cursor.execute("SELECT COUNT(*) FROM Tbl_history_models_import WHERE file_name = %s", (file_name,))
        if cursor.fetchone()[0] > 0:
            conn.close()
            return {'error': 'El archivo ya fue cargado previamente'}
        
        for index, row in df.iterrows():
            cursor.execute("""
                INSERT INTO Tbl_Students_Model (
                    age, gender_id, academic_level_id, country_id, avg_daily_used_hours, 
                    social_network_id, affects_academic_performance, sleep_hours_per_night, 
                    mental_health_score, relationship_status_id, conflicts_over_social_media, 
                    addicted_score, history_models_import_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1)
            """, (
                int(row.get('age', 0) or 0),
                int(row['gender_id']),
                int(row['academic_level_id']),
                int(row['country_id']),
                float(row.get('avgdailyusagehours', 0) or 0),
                int(row['social_network_id']),
                1 if str(row.get('affectsacademicperformance', '')).lower() in ['yes', 'si', '1', 'verdadero', 'true'] else 0,
                float(row.get('sleephourspernight', 0) or 0),
                float(row.get('mentalhealthscore', 0) or 0),
                int(row['relationship_status_id']),
                int(row.get('conflictsoversocialmedia', 0) or 0),
                float(row.get('addictedscore', 0) or 0)
            ))
        
        conn.commit()
        cursor.execute("INSERT INTO Tbl_history_models_import (file_name, isModelFirst, date_insert) VALUES (%s, 1, GETDATE())", (file_name,))
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': 'Archivo procesado e insertado correctamente'}
    
    except Exception as e:
        logger.error(f"Error procesando archivo: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return {'error': str(e)}