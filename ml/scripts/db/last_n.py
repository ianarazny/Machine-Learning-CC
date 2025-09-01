# last_n.py
import psycopg

CONN = dict(host="localhost", port=15432, dbname="metrics_db",
            user="postgres", password="postgres")

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
