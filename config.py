import os
from dotenv import load_dotenv
import pymssql
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class Config:
    SERVER = os.getenv('DB_SERVER')
    DATABASE = os.getenv('DB_NAME')
    USERNAME = os.getenv('DB_USER')
    PASSWORD = os.getenv('DB_PASSWORD')
    
    @staticmethod
    def get_connection():
        try:
            conn = pymssql.connect(
                server=Config.SERVER,
                user=Config.USERNAME,
                password=Config.PASSWORD,
                database=Config.DATABASE,
                port=1433,
                tds_version='7.2'
            )
            logger.info("Conexión con la base de datos establecida exitosamente")
            return conn
        except pymssql.Error as e:
            logger.error(f"Error de conexión a la base de datos: {str(e)}")
            raise Exception(f"Connection error: {str(e)}")