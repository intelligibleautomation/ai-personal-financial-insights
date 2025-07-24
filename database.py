import psycopg2
from config import Config
import logging

log = logging.getLogger("Database")


def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            host=Config.DB_HOST,
            port=Config.DB_PORT,
        )
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        log.info("✅ Successfully connected to PostgreSQL")
        return conn
    except psycopg2.OperationalError as e:
        log.error(f"❌ Database connection failed: {str(e)}")
        raise


def execute_query(query, params=None):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            conn.commit()
            return True
    except psycopg2.Error as e:
        log.error(f"❌ Query execution failed: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
