import psycopg
import os
from dotenv import load_dotenv
load_dotenv()


CONN = dict(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

qry = """
SELECT "timestamp", loadavg_1min, eno1_rx_bytes, eno1_tx_bytes
FROM public.sys_metrics
WHERE "timestamp" >= NOW() - %s::interval
ORDER BY "timestamp" DESC
LIMIT %s
"""

with psycopg.connect(**CONN) as conn, conn.cursor() as cur:
    cur.execute(qry, ('15 minutes', 20))  # OK: '15 minutes' -> ::interval
    for row in cur.fetchall():
        print(row)
