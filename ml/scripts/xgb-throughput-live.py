import os, time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import joblib
from pathlib import Path

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
THR_UNITS   = os.getenv("THR_UNITS", "mbps")

THR_EXPR = "throughput * 1e6" if THR_UNITS.lower() == "mbps" else "throughput"

# --- Features alineadas al notebook ---
FEATURES = [
    "throughput_bps_t","pps_t",
    "thr_lag1","thr_lag2","thr_lag3","thr_lag5",
    "pps_lag1","pps_lag2","pps_lag3","pps_lag5",
    "thr_ma_5","thr_std_5","pps_ma_5","thr_slope5"
]

# ================== ESTADO POR FLUJO ==================
@dataclass
class FlowState:
    thr: deque = field(default_factory=lambda: deque(maxlen=5))   # throughput_bps_t
    pps: deque = field(default_factory=lambda: deque(maxlen=5))   # pps_t
    ts : deque = field(default_factory=lambda: deque(maxlen=5))
    last_packets: int | None = None
    last_ts: pd.Timestamp | None = None

    def update_from_row(self, ts, throughput_bps_t, packets):
        """Actualiza pps desde 'packets' acumulativo usando deltat para robustez."""
        ts = pd.Timestamp(ts) 
        if self.last_packets is not None and self.last_ts is not None:
            dt_s = (ts - self.last_ts).total_seconds()
            if dt_s > 0 and packets is not None:
                dp = max(packets - self.last_packets, 0)
                pps_t = dp / dt_s
            else:
                pps_t = None
        else:
            pps_t = None

        self.last_packets = packets
        self.last_ts = ts

        # Guardar valores
        self.ts.append(ts)
        self.thr.append(float(throughput_bps_t) if throughput_bps_t is not None else 0.0)
        self.pps.append(float(pps_t) if pps_t is not None else np.nan)

    def feature_vector(self):
        """Construye el vector FEATURES si hay suficiente historia; si no, devuelve None."""
        if len(self.thr) < 5 or len(self.pps) < 5:
            return None

        thr_arr = np.array(self.thr, dtype=float)
        pps_arr = np.array(self.pps, dtype=float)

        # lags (1,2,3,5) → posiciones desde el final
        thr_lag1 = thr_arr[-2]
        thr_lag2 = thr_arr[-3]
        thr_lag3 = thr_arr[-4]
        thr_lag5 = thr_arr[0]

        pps_lag1 = pps_arr[-2]
        pps_lag2 = pps_arr[-3]
        pps_lag3 = pps_arr[-4]
        pps_lag5 = pps_arr[0]

        # ventanas sobre TODO el buffer (5 seg)
        thr_ma_5  = float(np.nanmean(thr_arr))
        thr_std_5 = float(np.nanstd(thr_arr, ddof=1)) if np.isfinite(thr_arr).sum() >= 2 else 0.0
        pps_ma_5  = float(np.nanmean(pps_arr))

        # pendiente (slope) en 5 s: diff(5)/5
        thr_slope5 = (thr_arr[-1] - thr_arr[0]) / 5.0

        current = {
            "throughput_bps_t": thr_arr[-1],
            "pps_t":            pps_arr[-1],
            "thr_lag1": thr_lag1, "thr_lag2": thr_lag2, "thr_lag3": thr_lag3, "thr_lag5": thr_lag5,
            "pps_lag1": pps_lag1, "pps_lag2": pps_lag2, "pps_lag3": pps_lag3, "pps_lag5": pps_lag5,
            "thr_ma_5": thr_ma_5, "thr_std_5": thr_std_5, "pps_ma_5": pps_ma_5, "thr_slope5": thr_slope5,
        }

        # Respeta el orden de FEATURES
        return np.array([current[k] for k in FEATURES], dtype=float)

# ================== HELPERS SQL ==================
def fetch_rows_since(since_ts: pd.Timestamp, limit=5000):
    """
    Trae filas nuevas de flow_metrics_logs desde since_ts (exclusivo), ordenadas por ts.
    Sólo lo necesario para features: ts, 5-tupla, throughput, packets.
    """
    sql = text(f"""
    SELECT
    ts,
    src_ip, dst_ip, src_port, dst_port, protocol,
    {THR_EXPR}::double precision AS throughput_bps_t,
    packets::bigint AS packets,
    bytes::bigint   AS bytes,
    delta_ns::bigint AS delta_ns
    FROM {FLOW_TABLE}
    WHERE ts > :since
    ORDER BY ts
    LIMIT :lim;
    """)
    with ENGINE.connect() as con:
        df = pd.read_sql(sql, con, params={"since": since_ts, "lim": limit})
    # Asegurar tz-aware
    if not isinstance(df["ts"].dtype, pd.DatetimeTZDtype):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # flow_id y pps_t
    df["flow_id"] = (
        df["src_ip"].astype(str) + ":" + df["src_port"].astype(str) + ">" +
        df["dst_ip"].astype(str) + ":" + df["dst_port"].astype(str) + "/" +
        df["protocol"].astype(str)
    )

    # dif de paquetes por flujo y por muestra, protegido contra negativos/0
    dpkts = df.groupby("flow_id", sort=False)["packets"].diff().clip(lower=0)
    dt_ns = df["delta_ns"].replace(0, pd.NA)
    df["pps_t"] = (dpkts * 1e9 / dt_ns).fillna(0).astype(float)

    return df

def save_prediction(row, yhat, thr_now, pps_now):
    if not SAVE_TO_DB:
        return
    sql = text("""
        INSERT INTO pred_bps_h1s
        (ts, src_ip, dst_ip, src_port, dst_port, protocol,
         throughput_bps_t, pps_t, yhat_bps_next_1s)
        VALUES (:ts, :src_ip, :dst_ip, :src_port, :dst_port, :protocol,
                :thr_now, :pps_now, :yhat)
        -- Si agregaste UNIQUE(ts,5-tupla), podés habilitar el upsert:
        -- ON CONFLICT (ts, src_ip, dst_ip, src_port, dst_port, protocol)
        -- DO UPDATE SET
        --   throughput_bps_t = EXCLUDED.throughput_bps_t,
        --   pps_t            = EXCLUDED.pps_t,
        --   yhat_bps_next_1s = EXCLUDED.yhat_bps_next_1s
    """)
    params = {
        "ts": row["ts"],
        "src_ip": str(row["src_ip"]),           # Postgres castea string→inet
        "dst_ip": str(row["dst_ip"]),
        "src_port": int(row["src_port"]),
        "dst_port": int(row["dst_port"]),
        "protocol": int(row["protocol"]),
        "thr_now": float(thr_now),
        "pps_now": float(pps_now) if np.isfinite(pps_now) else 0.0,
        "yhat": float(max(yhat, 0.0)),          # clamp a 0 si prefieres evitar negativos
    }
    with ENGINE.begin() as con:
        con.execute(sql, params)


# ================== MAIN LOOP ==================
def main():
    print("Cargando modelo:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    # Estado por flujo
    states: dict[str, FlowState] = defaultdict(FlowState)

    # Warmup: arrancamos unos segundos atrás para llenar buffers
    now = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
    since_ts = now - pd.Timedelta(seconds=WARMUP_SEC)

    print(f"Arrancando desde {since_ts.isoformat()} (warmup {WARMUP_SEC}s).")
    last_seen = since_ts

    while True:
        try:
            df = fetch_rows_since(last_seen)
            if not df.empty:
                # procesar en orden cronológico
                for _, r in df.iterrows():
                    flow_id = f"{str(r['src_ip'])}:{int(r['src_port'])}>{str(r['dst_ip'])}:{int(r['dst_port'])}/{int(r['protocol'])}"
                    st = states[flow_id]
                    # actualizar estado con la muestra actual
                    st.update_from_row(r["ts"], r["throughput_bps_t"], r["packets"])

                    # construir features si hay buffer suficiente
                    x = st.feature_vector()
                    if x is not None and np.isfinite(x).all():
                        yhat = float(model.predict(x.reshape(1, -1))[0])

                        # log consola (en Mbps)
                        print(f"{r['ts']} {flow_id}  thr_now={st.thr[-1]/1e6:.2f} Mbps  "
                              f"pps={st.pps[-1]:.0f}  yhat(+1s)={yhat/1e6:.2f} Mbps")

                        thr_now = st.thr[-1]
                        pps_now = st.pps[-1]
                        save_prediction(r, yhat, thr_now, pps_now)


                    # avanzar puntero
                    last_seen = max(last_seen, r["ts"])
            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            print("Detenido por usuario.")
            break
        except Exception as e:
            # No te quedes mudo si algo se rompe: logueá y seguí
            print("Error en loop:", repr(e))
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
