import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

# =========================
# 0. CONFIGURACIÓN GLOBAL
# =========================
SEED = 42
HORIZON_SECONDS = 1             # predecir pérdida a +1s
RESAMPLE_RULE = '1s'            # trabajamos en rejilla de 1 segundo
MIN_EXPECTED_FOR_LOSS = 1e-6    # epsilon para división
TOP_N_PORTS = 20                # one-hot puertos frecuentes
ROLL_WINDOWS = [3, 5, 10]       # ventanas en segundos
LAGS = [1, 2, 3]                # lags en segundos
USE_OBSERVED_SERIES = 'pps'     # 'pps' o 'throughput' como serie observada base para el proxy

# ================
# 1. PREPROCESO
# ================
def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena, filtra TCP, garantiza tipos y parsea timestamp.
    Asume contadores cumulativos por flow_let (packets/bytes...).
    """
    df = df.copy()

    # Parseo de timestamp y orden temporal
    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = pd.to_datetime(df['ts'], utc=True, errors='coerce')
    df = df.dropna(subset=['ts']).sort_values('ts')

    # Filtrar TCP si te interesa solo TCP (recomendado para pérdida)
    df = df[df['protocol'] == 6].copy()

    # Asegurar columnas presentes (llenar faltantes si corresponde)
    for col in ['packets', 'bytes', 'delta_ns', 'throughput',
                'fin', 'syn', 'rst', 'psh', 'ack', 'urg']:
        if col not in df.columns:
            df[col] = 0

    # Si hay NAs en contadores/flags, los rellenamos a 0
    df[['packets','bytes','delta_ns','throughput','fin','syn','rst','psh','ack','urg']] = \
        df[['packets','bytes','delta_ns','throughput','fin','syn','rst','psh','ack','urg']].fillna(0)

    return df


def _flow_id_cols():
    return ['src_ip','dst_ip','src_port','dst_port','protocol','flow_let']


def to_timegrid_1s(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rejilla de 1s por flujo (5-tuple + protocol + flow_let).
    Calcula tasas por segundo a partir de contadores cumulativos:
    - pps (paquetes/s)
    - bps (bits/s) a partir de bytes
    - flags por s (fin/syn/rst/psh/ack/urg)
    """
    df = df.copy()
    gcols = _flow_id_cols()

    keep = gcols + ['ts','packets','bytes','delta_ns','throughput','fin','syn','rst','psh','ack','urg']
    df = df[keep].copy()

    # Garantizar ts válido y orden
    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = pd.to_datetime(df['ts'], utc=True, errors='coerce')
    df = df.dropna(subset=['ts']).sort_values('ts')

    # Trabajamos por flujo
    df = df.set_index('ts').sort_index()

    def _resample_flow(flow_df):
        # resample a 1s sobre el índice datetime
        flow_1s = flow_df.resample(RESAMPLE_RULE).last()

        # Forward-fill en contadores y señales
        cols_ffill = ['packets','bytes','throughput','fin','syn','rst','psh','ack','urg']
        flow_1s[cols_ffill] = flow_1s[cols_ffill].ffill()

        # dt real (segundos)
        dt = flow_1s.index.to_series().diff().dt.total_seconds().fillna(0.0)

        # Deltas no negativas
        for c in ['packets','bytes','fin','syn','rst','psh','ack','urg']:
            flow_1s[f'd_{c}'] = flow_1s[c].diff().clip(lower=0).fillna(0)

        denom = dt.replace(0, np.nan)
        flow_1s['pps'] = (flow_1s['d_packets'] / denom).fillna(0)
        flow_1s['bps'] = (flow_1s['d_bytes'] * 8 / denom).fillna(0)

        for c in ['fin','syn','rst','psh','ack','urg']:
            flow_1s[f'{c}_ps'] = (flow_1s[f'd_{c}'] / denom).fillna(0)

        flow_1s['bytes_per_pkt'] = np.where(flow_1s['d_packets'] > 0,
                                            flow_1s['d_bytes'] / flow_1s['d_packets'], 0.0)

        first_ts = flow_1s.index.min()
        flow_1s['flowlet_age_s'] = (flow_1s.index - first_ts).total_seconds()

        return flow_1s

    grid = (df.groupby(_flow_id_cols(), group_keys=True, sort=False, as_index=True).apply(_resample_flow, include_groups=False))


    # Reconstrucción simple de columnas
    grid.index.set_names(_flow_id_cols() + ['ts'], inplace=True)
    grid = grid.reset_index()

    # Orden final
    grid = grid.sort_values(['src_ip','dst_ip','src_port','dst_port','protocol','flow_let','ts'])
    return grid



# ===================================
# 2. CÁLCULO DE PACKET LOSS (PROXY)
# ===================================
def compute_loss_proxy(df1s: pd.DataFrame,
                       observed_col: str = USE_OBSERVED_SERIES,
                       ema_span: int = 5) -> pd.DataFrame:
    """
    Calcula el proxy de pérdida por 1s:
    loss_t = clip((expected_t - observed_t)/expected_t, 0,1)
    expected_t = EMA( observed ) usando sólo pasado (ajustamos con shift).

    Luego define y_t = loss_{t+1} (target a +1s).
    """
    df = df1s.copy()
    gcols = _flow_id_cols()

    # Elegir serie observada base (pps o throughput)
    if observed_col not in df.columns:
        raise ValueError(f"'{observed_col}' no existe en df. Opciones: 'pps' o 'throughput'.")

    # EMA causal por flujo
    def _expected(x):
        # EMA solo con pasado: primero shift 1, luego ewm
        return x.shift(1).ewm(span=ema_span, adjust=False, min_periods=3).mean()

    df['expected_obs'] = df.groupby(gcols, group_keys=False)[observed_col].apply(_expected)

    # loss proxy (0..1)
    df['packet_loss'] = np.clip(
        (df['expected_obs'] - df[observed_col]) / (df['expected_obs'].abs() + MIN_EXPECTED_FOR_LOSS),
        0.0, 1.0
    )

    # Para expected muy chico, setear 0 (evita artefactos)
    df.loc[df['expected_obs'].abs() < 1e-9, 'packet_loss'] = 0.0

    # Objetivo a +1s
    df['y_future_loss'] = df.groupby(gcols, group_keys=False)['packet_loss'].shift(-HORIZON_SECONDS)

    return df


# ===========================
# 3. FEATURE ENGINEERING
# ===========================
def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva features temporales puras a partir de 'ts'.
    """
    x = df.copy()
    x['hour'] = x['ts'].dt.hour
    x['dow'] = x['ts'].dt.dayofweek
    x['minute'] = x['ts'].dt.minute
    x['second'] = x['ts'].dt.second
    return x


def encode_network_categoricals(df: pd.DataFrame, top_n_ports=TOP_N_PORTS):
    """
    - Puertos frecuentes (src/dst): one-hot para top-N.
    - Resto -> 'other_src_port'/'other_dst_port' binaria.
    - IPs: frequency encoding (apariciones relativas).
    """
    x = df.copy()

    # Top-N puertos fuente/destino
    top_src = x['src_port'].value_counts().head(top_n_ports).index
    top_dst = x['dst_port'].value_counts().head(top_n_ports).index

    for p in top_src:
        x[f'src_port_{p}'] = (x['src_port'] == p).astype(int)
    x['src_port_other'] = (~x['src_port'].isin(top_src)).astype(int)

    for p in top_dst:
        x[f'dst_port_{p}'] = (x['dst_port'] == p).astype(int)
    x['dst_port_other'] = (~x['dst_port'].isin(top_dst)).astype(int)

    # Frequency encoding para IPs
    src_freq = x['src_ip'].value_counts(normalize=True)
    dst_freq = x['dst_ip'].value_counts(normalize=True)
    x['src_ip_freq'] = x['src_ip'].map(src_freq).fillna(0.0)
    x['dst_ip_freq'] = x['dst_ip'].map(dst_freq).fillna(0.0)

    # IP privada (heurística)
    def _is_private(ip):
        try:
            s = str(ip)
            return s.startswith('10.') or s.startswith('192.168.') or s.startswith('172.16.')
        except:
            return False
    x['src_private'] = x['src_ip'].apply(_is_private).astype(int)
    x['dst_private'] = x['dst_ip'].apply(_is_private).astype(int)

    return x


def make_lag_rolling_features(df: pd.DataFrame,
                              lag_cols=('pps','bps','throughput','bytes_per_pkt',
                                       'fin_ps','syn_ps','rst_ps','psh_ps','ack_ps','urg_ps'),
                              lags=LAGS, roll_windows=ROLL_WINDOWS) -> pd.DataFrame:
    """
    Construye lags y rolling stats causales por flujo.
    """
    x = df.copy()
    gcols = _flow_id_cols()

    # Lags
    for col in lag_cols:
        if col not in x.columns:
            continue
        for L in lags:
            x[f'{col}_lag{L}'] = x.groupby(gcols, group_keys=False)[col].shift(L)

    # Rolling stats (media, std, min, max)
    def _roll_group(s, w):
        # Rolling causal cerrado a la izquierda:
        return s.shift(1).rolling(window=w, min_periods=max(2, w//2))

    for col in lag_cols:
        if col not in x.columns:
            continue
        for w in roll_windows:
            r = x.groupby(gcols, group_keys=False)[col].apply(lambda s: _roll_group(s, w))
            x[f'{col}_roll{w}_mean'] = r.mean()
            x[f'{col}_roll{w}_std']  = r.std()
            x[f'{col}_roll{w}_min']  = r.min()
            x[f'{col}_roll{w}_max']  = r.max()

    return x


def assemble_dataset(df_with_loss: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Ensambla X (features), y (target) y un df de contexto para análisis.
    Filtra filas donde y_future_loss es NaN (bordes de cada flowlet).
    """
    x = df_with_loss.copy()
    x = build_time_features(x)
    x = encode_network_categoricals(x)
    x = make_lag_rolling_features(x)

    # Columnas numéricas útiles (evitar IDs crudos con cardinalidad enorme)
    feature_cols = [
        # tasas base
        'pps','bps','throughput','bytes_per_pkt','flowlet_age_s',
        'fin_ps','syn_ps','rst_ps','psh_ps','ack_ps','urg_ps',
        # temporales
        'hour','dow','minute','second',
        # encoding de red
        'src_ip_freq','dst_ip_freq','src_private','dst_private',
    ]
    # Puertos one-hot
    feature_cols += [c for c in x.columns if c.startswith('src_port_')]
    feature_cols += [c for c in x.columns if c.startswith('dst_port_')]

    # Lags y rolling
    feature_cols += [c for c in x.columns if any(c.endswith(f'_lag{L}') for L in LAGS)]
    feature_cols += [c for c in x.columns if any(f'_roll{w}_' in c for w in ROLL_WINDOWS)]

    # Target y filtrado de NaNs
    y = x['y_future_loss']
    valid = y.notna()

    X = x.loc[valid, feature_cols].fillna(0.0)
    y = y.loc[valid].astype(float)
    ctx = x.loc[valid, _flow_id_cols() + ['ts','packet_loss','y_future_loss']]

    return X, y, ctx


# ==============================
# 4. ENTRENAMIENTO Y TUNING
# ==============================
def time_series_split_eval(X, y, n_splits=5, random_state=SEED):
    """
    TimeSeriesSplit + RandomizedSearchCV para XGBRegressor.
    Maneja desbalance dando mayor peso a y>0.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Modelo base
    xgb = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=600,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        tree_method='hist'  # 'gpu_hist' si tenés GPU
    )

    # Espacio de búsqueda razonable
    param_distributions = {
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 8),
        'reg_alpha': loguniform(1e-4, 1e-1),
        'reg_lambda': loguniform(1e-3, 1.0),
        'gamma': loguniform(1e-4, 1e-1),
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'learning_rate': [0.03, 0.05, 0.08],
        'n_estimators': [400, 600, 800]
    }

    # Pesos para desbalance: más peso si y>0
    sample_weight = np.where(y.values > 0, 3.0, 1.0)

    # Pipeline con estandarización ligera (opcional)
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),  # sparse-friendly
        ('model', xgb)
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions={
            # Para acceder a hiperparámetros internos del estimador en pipeline:
            'model__' + k: v for k, v in param_distributions.items()
        },
        n_iter=30,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    search.fit(X, y, model__sample_weight=sample_weight)
    return search


# =========================
# 5. EVALUACIÓN COMPLETA
# =========================
def evaluate_model(best_pipe, X, y, ctx, title='XGBRegressor packet_loss @+1s'):
    """
    Reporta MAE, RMSE, R2, MAPE y gráficos de:
    - Histograma de y y y_pred
    - y vs y_pred
    - Residuales en el tiempo (muestra)
    """
    y_pred = best_pipe.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    mape = (np.abs((y - y_pred) / (np.maximum(np.abs(y), 1e-8)))).mean()

    print(f'[{title}]')
    print(f'MAE : {mae:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'R²  : {r2:.4f}')
    print(f'MAPE: {mape:.6f}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    sns.histplot(y, kde=False, ax=axes[0])
    sns.histplot(y_pred, kde=False, ax=axes[0], alpha=0.5)
    axes[0].set_title('Distribución y (real) vs y_pred')

    axes[1].scatter(y, y_pred, s=5, alpha=0.4)
    axes[1].set_xlabel('y (real)')
    axes[1].set_ylabel('y_pred')
    axes[1].set_title('y vs y_pred')

    # Residuales temporalmente (muestra 50k)
    resid = y - y_pred
    sample = resid.sample(min(50000, len(resid)), random_state=SEED)
    axes[2].plot(sample.values)
    axes[2].set_title('Residuales (muestra)')
    plt.tight_layout()
    plt.show()

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}, y_pred


# ============================
# 6. ANÁLISIS EXPLORATORIO
# ============================
def exploratory_analysis(df_with_loss):
    """
    EDA básico:
    - Stats y distribución del packet_loss proxy
    - Correlaciones base
    - Agregados por hora/día
    - Por protocolo/puertos/IPs (para TCP ya filtrado)
    - Outliers simples
    """
    df = df_with_loss.copy()

    print('Estadísticas packet_loss (proxy):')
    print(df['packet_loss'].describe(percentiles=[.5,.9,.95,.99]))

    plt.figure(figsize=(6,4))
    sns.histplot(df['packet_loss'], bins=50)
    plt.title('Distribución de packet_loss (proxy)')
    plt.show()

    # Por hora del día
    df['hour'] = df['ts'].dt.hour
    hourly = df.groupby('hour')['packet_loss'].mean()
    plt.figure(figsize=(8,4))
    hourly.plot(kind='bar')
    plt.title('Promedio de packet_loss por hora')
    plt.show()

    # Correlaciones con algunas señales base
    corr_cols = ['packet_loss','pps','bps','throughput','bytes_per_pkt',
                 'fin_ps','syn_ps','rst_ps','psh_ps','ack_ps','urg_ps','flowlet_age_s']
    corr = df[corr_cols].corr()
    plt.figure(figsize=(9,7))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlaciones')
    plt.show()

    # Puertos más asociados a pérdida (proxy)
    top_src = df['src_port'].value_counts().head(10).index
    print('Packet loss promedio por src_port (top10):')
    print(df[df['src_port'].isin(top_src)].groupby('src_port')['packet_loss'].mean().sort_values(ascending=False).head(10))

    # Outliers simples (picos)
    thr = df['packet_loss'].quantile(0.99)
    outliers = df[df['packet_loss']>=thr]
    print(f'Outliers (≥ p99={thr:.3f}): {len(outliers)} filas. Ejemplo:')
    print(outliers.head())


# ============================
# 7. PIPELINE COMPLETO
# ============================
def run_pipeline(df_raw: pd.DataFrame):
    # 1) Preproceso base
    df0 = preprocess_raw(df_raw)

    # 2) Rejilla 1s por flowlet + tasas
    df1s = to_timegrid_1s(df0)

    # 3) Objetivo de pérdida (proxy) y desplazamiento a +1s
    df_loss = compute_loss_proxy(df1s, observed_col=USE_OBSERVED_SERIES, ema_span=5)

    # 4) EDA (opcional, comentar si no querés plots)
    exploratory_analysis(df_loss)

    # 5) Ensamble dataset (X, y)
    X, y, ctx = assemble_dataset(df_loss)

    # 6) Entrenamiento con validación temporal + tuning
    search = time_series_split_eval(X, y, n_splits=5, random_state=SEED)
    print('Mejor score (MAE negativo):', search.best_score_)
    print('Mejores hiperparámetros:', search.best_params_)

    best_pipe = search.best_estimator_

    # 7) Evaluación global post-tuning
    metrics, y_pred = evaluate_model(best_pipe, X, y, ctx, title='XGB packet_loss_proxy @+1s')

    # 8) Importancia de features
    # Extraemos el modelo interno
    model = best_pipe.named_steps['model']
    # NB: si escalaste, no afecta importancia en XGB
    importances = model.feature_importances_
    fi = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
    print('Top 20 features:')
    print(fi.head(20))

    plt.figure(figsize=(8,10))
    sns.barplot(data=fi.head(20), x='importance', y='feature')
    plt.title('Importancia de features (Top 20)')
    plt.tight_layout()
    plt.show()

    return best_pipe, metrics, fi, (X, y, ctx)


# ============================
# 8. EJEMPLO DE USO
# ============================
if __name__ == '__main__':
    # Carga de datos:
    # df_raw = pd.read_csv('flow_metrics_logs.csv', parse_dates=['ts'])  # o desde SQL
    # Con SQLAlchemy/Psycopg, traer columnas relevantes ordenadas por ts.

    HOST = os.getenv("DB_HOST")
    PORT = os.getenv("DB_PORT")
    DB   = os.getenv("DB_NAME")
    USER = os.getenv("DB_USER")
    PASS = os.getenv("DB_PASSWORD")

    engine = create_engine(f"postgresql+psycopg2://{USER}:{PASS}@{HOST}:{PORT}/{DB}")
    Q = """
    SELECT ts,pid,src_ip,dst_ip,src_port,dst_port,protocol,flow_let,last_timestamp_ns,delta_ns,packets,bytes,fin,syn,rst,psh,ack,urg,throughput
    FROM flow_metrics_logs
    """
    df = pd.read_sql(Q, engine)

    best_model, metrics, feature_importance, data_ctx = run_pipeline(df)
    pass
