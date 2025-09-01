import psycopg
import os
from dotenv import load_dotenv
load_dotenv()


conn = dict(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

with conn, conn.cursor() as cur:
    cur.execute("""
        SELECT datname, pg_get_userbyid(datdba) AS owner
        FROM pg_database
        WHERE datistemplate = false
        ORDER BY datname;
    """)
    for row in cur.fetchall():
        print(row)


