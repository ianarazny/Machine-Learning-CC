import psycopg
import os
from dotenv import load_dotenv
load_dotenv()

QUERY = """
SELECT ts_1s AS ts, src_ip, dst_ip, src_port, dst_port, protocol,
       throughput_bps_t, pps_t, y_bps_next_1s
FROM training_bps_h1s
WHERE ts_1s IS NOT NULL
ORDER BY ts_1s;
"""

CONN = dict(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

with psycopg.connect(**CONN) as conn, conn.cursor() as cur:
    # 1) Info básica de la sesión
    cur.execute("SELECT version(), current_database(), current_user;")
    version, db, usr = cur.fetchone()
    print("version:", version)
    print("db/user:", db, usr)

    # 2) Listar tablas "de usuario"
    cur.execute("""
        SELECT schemaname, tablename
        FROM pg_catalog.pg_tables
        WHERE schemaname NOT IN ('pg_catalog','information_schema')
        ORDER BY 1,2
    """)
    tables = cur.fetchall()
    print("\nTablas:")
    for s, t in tables:
        print(f" - {s}.{t}")

    cur.execute("SELECT to_regclass('public.sys_metrics')")
    if cur.fetchone()[0]:
        print("\npublic.sys_metrics existe")

        # columnas
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='sys_metrics'
            ORDER BY ordinal_position
        """)
        print("Columnas sys_metrics:")
        for col, typ in cur.fetchall():
            print(f" - {col}: {typ}")

        # conteo y 5 filas de muestra
        cur.execute("SELECT COUNT(*) FROM public.sys_metrics;")
        print("Filas en sys_metrics:", cur.fetchone()[0])

        cur.execute("SELECT * FROM public.sys_metrics LIMIT 5;")
        sample = cur.fetchall()
        print("\nMuestra (5 filas):")
        for row in sample:
            print(row)
    else:
        print("\nNo se encontró public.sys_metrics (omitido muestreo)")
