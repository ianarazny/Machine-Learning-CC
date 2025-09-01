# discover_dbs.py
import psycopg

conn = psycopg.connect(
    host="localhost", port=15432,
    dbname="metrics_db", user="postgres", password="postgres"
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


