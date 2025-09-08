# rt_pred_bps_h1s.py
import os, time
from collections import deque, defaultdict
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import joblib


import os
from pathlib import Path
from urllib.parse import quote_plus
from sqlalchemy import create_engine

# --- DB (desde .env / entorno) ---
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "15432")
DB_NAME     = os.getenv("DB_NAME", "metrics_db")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_SCHEMA   = os.getenv("DB_SCHEMA", "public")   # por si usas un schema distinto a public

ENGINE = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# También versión psycopg2 directa (pandas.read_sql/otros):
CONN = dict(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
)

# --- Rutas y modelo ---
MODEL_NAME = os.getenv("MODEL_NAME", "xgb_bps_h1s_v3.joblib")
REPO_ROOT  = Path(__file__).resolve().parents[1]   # .../ml
MODEL_PATH = (REPO_ROOT / "models" / MODEL_NAME)
print("Cargando modelo:", MODEL_PATH)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No existe el modelo en: {MODEL_PATH}")

# --- Polling / runtime ---
POLL_SECONDS  = int(os.getenv("POLL_SECONDS", "1"))       # frecuencia de polling
WARMUP_SEC    = int(os.getenv("WARMUP_SEC",   "10"))      # historial para llenar buffers
SAVE_TO_DB    = os.getenv("SAVE_TO_DB", "true").lower() == "true"
MODEL_VERSION = os.getenv("MODEL_VERSION", "xgb_bps_h1s_v1")

# --- Tablas/Vistas (configurables por env) ---
TRAIN_TABLE = os.getenv("TRAIN_TABLE", f"{DB_SCHEMA}.training_bps_h1s")
FLOW_TABLE  = os.getenv("FLOW_TABLE",  f"{DB_SCHEMA}.flow_metrics_logs")  # por si también lo usas en vivo

# --- Features alineadas al notebook ---
FEATURES = [
    "throughput_bps_t","pps_t",
    "thr_lag1","thr_lag2","thr_lag3","thr_lag5",
    "pps_lag1","pps_lag2","pps_lag3","pps_lag5",
    "thr_ma_5","thr_std_5","pps_ma_5","thr_slope5"
]
class FlowState:
    def __init__(self):
        self.thr = deque(maxlen=5)
        self.pps = deque(maxlen=5)
        self.ts  = deque(maxlen=5)
        self.last_packets = None
        self.last_ts = None

    def update(self, ts, throughput_bps_t, packets):
        ts = pd.to_datetime(ts, utc=True)
        # pps desde contador acumulativo
        if self.last_packets is not None and self.last_ts is not None:
            dt = (ts - self.last_ts).total_seconds()
            pps_t = (max(packets - self.last_packets, 0) / dt) if dt > 0 else np.nan
        else:
            pps_t = np.nan
        self.last_packets = packets
        self.last_ts = ts

        self.ts.append(ts)
        self.thr.append(float(throughput_bps_t) if throughput_bps_t is not None else 0.0)
        self.pps.append(float(pps_t))

    def features(self):
        if len(self.thr) < 5 or len(self.pps) < 5:
            return None
        thr = np.array(self.thr, dtype=float)
        pps = np.array(self.pps, dtype=float)

        feats = {
            "throughput_bps_t": thr[-1],
            "pps_t":            pps[-1],
            "thr_lag1": thr[-2], "thr_lag2": thr[-3], "thr_lag3": thr[-4], "thr_lag5": thr[0],
            "pps_lag1": pps[-2], "pps_lag2": pps[-3], "pps_lag3": pps[-4], "pps_lag5": pps[0],
            "thr_ma_5":  float(np.nanmean(thr)),
            "thr_std_5": float(np.nanstd(thr, ddof=1)) if np.isfinite(thr).sum() >= 2 else 0.0,
            "pps_ma_5":  float(np.nanmean(pps)),
            "thr_slope5": (thr[-1] - thr[0]) / 5.0
        }
        return np.array([feats[k] for k in FEATURES], dtype=float), feats  # también devuelvo curr vals

def last_processed_ts():
    sql = text(f"SELECT max(ts) AS mx FROM {TABLE_PRED};")
    with ENGINE.connect() as con:
        r = con.execute(sql).mappings().first()
    return pd.to_datetime(r["mx"], utc=True) if r and r["mx"] else None

def fetch_rows_since(since_ts, limit=5000):
    sql = text("""
        SELECT
          ts,
          src_ip, dst_ip, src_port, dst_port, protocol,
          throughput::double precision AS throughput_bps_t,
          packets::bigint AS packets
        FROM flow_metrics_logs
        WHERE ts > :since
        ORDER BY ts
        LIMIT :lim;
    """)
    with ENGINE.connect() as con:
        df = pd.read_sql(sql, con, params={"since": since_ts, "lim": limit})
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df

def upsert_pred(row, feats_vals, yhat):
    """
    Inserta (o actualiza) una predicción para (ts, 5-tupla).
    Guardamos los valores actuales throughput_bps_t y pps_t para trazabilidad.
    """
    sql = text(f"""
        INSERT INTO {TABLE_PRED}
        (ts, src_ip, dst_ip, src_port, dst_port, protocol,
         throughput_bps_t, pps_t, yhat_bps_next_1s)
        VALUES (:ts, :src_ip, :dst_ip, :src_port, :dst_port, :protocol,
                :thr_now, :pps_now, :yhat)
        ON CONFLICT (ts, src_ip, dst_ip, src_port, dst_port, protocol)
        DO UPDATE SET
            throughput_bps_t = EXCLUDED.throughput_bps_t,
            pps_t            = EXCLUDED.pps_t,
            yhat_bps_next_1s = EXCLUDED.yhat_bps_next_1s;
    """)
    params = {
        "ts": row["ts"],
        "src_ip": str(row["src_ip"]),    # string castea a inet
        "dst_ip": str(row["dst_ip"]),
        "src_port": int(row["src_port"]),
        "dst_port": int(row["dst_port"]),
        "protocol": int(row["protocol"]),
        "thr_now": float(feats_vals["throughput_bps_t"]),
        "pps_now": float(feats_vals["pps_t"]) if np.isfinite(feats_vals["pps_t"]) else 0.0,
        "yhat": float(yhat)
    }
    with ENGINE.begin() as con:
        con.execute(sql, params)

def main():
    print("Cargando modelo:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    # Arrancar desde el último ts ya persistido (si hay), con warmup hacia atrás
    mx_ts = last_processed_ts()
    since = (mx_ts - pd.Timedelta(seconds=WARMUP_SEC)) if mx_ts is not None else (pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(seconds=WARMUP_SEC))
    print(f"Desde ts={since.isoformat()} (warmup {WARMUP_SEC}s).")

    states = defaultdict(FlowState)
    last_seen = since

    while True:
        try:
            df = fetch_rows_since(last_seen)
            if not df.empty:
                for _, r in df.iterrows():
                    fid = f"{str(r['src_ip'])}:{int(r['src_port'])}>{str(r['dst_ip'])}:{int(r['dst_port'])}/{int(r['protocol'])}"
                    st = states[fid]
                    st.update(r["ts"], r["throughput_bps_t"], r["packets"])

                    feats = st.features()
                    if feats is not None:
                        x, curr = feats
                        if np.isfinite(x).all():
                            yhat = float(model.predict(x.reshape(1,-1))[0])
                            # log rápido
                            print(f"{r['ts']} {fid}  thr={curr['throughput_bps_t']/1e6:.2f}Mb  "
                                  f"pps={curr['pps_t']:.0f}  yhat(+1s)={yhat/1e6:.2f}Mb")
                            # persistir
                            upsert_pred(r, curr, yhat)

                    last_seen = max(last_seen, r["ts"])

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("Detenido por usuario.")
            break
        except Exception as e:
            print("Error en loop:", repr(e))
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
