import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

from alpha_vantage.foreignexchange import ForeignExchange

# --- CONFIG STREAMLIT ---
st.set_page_config(page_title="Scanner Confluence Forex (AlphaVantage)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Alpha Vantage)")
st.markdown("*Utilisation de l'API Alpha Vantage pour les données de marché*")

# --- API KEY ---
try:
    AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
except KeyError:
    st.error("Erreur : Secret 'ALPHA_VANTAGE_API_KEY' manquant.")
    st.stop()

try:
    fx = ForeignExchange(key=AV_API_KEY, output_format='pandas')
    st.sidebar.success("Client Alpha Vantage initialisé.")
except Exception as e:
    st.sidebar.error(f"Erreur Alpha Vantage : {e}")
    st.stop()

# --- PAIRES FOREX ---
FOREX_PAIRS_AV = [
    ('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY'), ('USD', 'CHF'),
    ('AUD', 'USD'), ('USD', 'CAD'), ('NZD', 'USD'), ('EUR', 'JPY'),
    ('GBP', 'JPY'), ('EUR', 'GBP')
]

# --- INDICATEURS ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma_pine(dc, p=20):
    hl, sl = int(p / 2), int(np.sqrt(p))
    wma1 = dc.rolling(hl).apply(lambda x: np.dot(x, range(1, len(x)+1)) / np.sum(range(1, len(x)+1)), raw=True)
    wma2 = dc.rolling(p).apply(lambda x: np.dot(x, range(1, len(x)+1)) / np.sum(range(1, len(x)+1)), raw=True)
    diff = 2 * wma1 - wma2
    return diff.rolling(sl).apply(lambda x: np.dot(x, range(1, len(x)+1)) / np.sum(range(1, len(x)+1)), raw=True)

def rsi_pine(po4, p=10):
    d = po4.diff()
    g, l = d.clip(lower=0), -d.clip(upper=0)
    rs = rma(g, p) / rma(l, p).replace(0, 1e-9)
    return 100 - (100 / (1 + rs)).fillna(50)

def adx_pine(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, p)
    up_move = h.diff()
    down_move = l.shift() - l
    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    mdm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    pdi = 100 * (rma(pd.Series(pdm), p) / atr.replace(0, 1e-9))
    mdi = 100 * (rma(pd.Series(mdm), p) / atr.replace(0, 1e-9))
    dx = 100 * (abs(pdi - mdi) / (pdi + mdi).replace(0, 1e-9))
    return rma(dx, p).fillna(0)

def heiken_ashi_pine(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    if not df.empty:
        ha_open.iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    return ha_open, ha_close

# --- API ALPHA VANTAGE ---
@st.cache_data(ttl=3600)
def get_data_av(from_currency: str, to_currency: str, interval: str = '60min', output_size: str = 'compact'):
    try:
        df, meta = fx.get_currency_exchange_intraday(
            from_symbol=from_currency,
            to_symbol=to_currency,
            interval=interval,
            outputsize=output_size
        )
        df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close'
        }, inplace=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()  # ordre chronologique
        return df.dropna()
    except Exception as e:
        st.error(f"Erreur chargement données AV {from_currency}/{to_currency} : {str(e)}")
        return None

# --- AFFICHAGE ---
with st.sidebar:
    selected_pairs = st.multiselect("Sélectionnez des paires", FOREX_PAIRS_AV, default=[('EUR', 'USD'), ('GBP', 'USD')])
    go = st.button("Analyser")

if go:
    for fc, tc in selected_pairs:
        df = get_data_av(fc, tc)
        if df is not None and not df.empty:
            st.subheader(f"{fc}/{tc}")
            st.line_chart(df['Close'])
            # Tu peux insérer ici le calcul des signaux (HMA, RSI, ADX, etc.)
        else:
            st.warning(f"Pas de données disponibles pour {fc}/{tc}")

