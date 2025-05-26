import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback # Pour le traceback détaillé dans les logs

# Import pour Alpaca
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame # TimeFrame est une Enum

# --- Configuration de la page et titre ---
st.set_page_config(page_title="Scanner Confluence Forex (Alpaca)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Alpaca)")
st.markdown("*Utilisation de l'API Alpaca pour les données de marché*")

# --- Récupération sécurisée des Clés API Alpaca depuis Streamlit Secrets ---
api = None # Déclarer api ici pour qu'il soit dans le scope global du module
try:
    API_KEY = st.secrets["ALPACA_API_KEY"]
    API_SECRET = st.secrets["ALPACA_SECRET_KEY"]
    BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets") 
except KeyError as e:
    st.error(f"Erreur: La clé secrète Streamlit '{e.args[0]}' n'est pas définie. Veuillez configurer vos secrets Alpaca.")
    st.stop()
except Exception as e: 
    st.error(f"Une erreur s'est produite lors de la lecture des secrets Streamlit: {e}")
    st.stop()

# Initialiser l'API Alpaca
try:
    api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL)
    account = api.get_account()
    st.sidebar.success(f"Connecté au compte Alpaca (Paper): {account.account_number}")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation ou de la connexion à l'API Alpaca: {e}")
    st.sidebar.error("Échec de la connexion à Alpaca.")
    api = None # S'assurer que api est None si l'initialisation échoue
    # st.stop() # On pourrait arrêter ici, ou laisser l'app se charger et échouer au scan

# --- Liste des paires Forex (Format Alpaca) ---
FOREX_PAIRS_ALPACA = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
    'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY',
    'GBPJPY', 'EURGBP'
    # 'XAUUSD' # Tu peux ajouter l'or si tu confirmes son symbole sur Alpaca
]

# --- Mapping pour les Timeframes Alpaca ---
TIMEFRAME_MAP_ALPACA = {
    "1Min": TimeFrame.Minute,
    "5Min": TimeFrame(5, tradeapi.rest.TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, tradeapi.rest.TimeFrameUnit.Minute),
    "1H": TimeFrame.Hour,
    "4H": TimeFrame(4, tradeapi.rest.TimeFrameUnit.Hour),
    "1D": TimeFrame.Day
}

# --- Fonctions d'indicateurs techniques (INCHANGÉES) ---
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()
def rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()
def hull_ma_pine(data_close, period=20):
    half_length = int(period / 2); sqrt_length = int(np.sqrt(period))
    wma_half_period = data_close.rolling(window=half_length).apply(lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    wma_full_period = data_close.rolling(window=period).apply(lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    diff_wma = 2 * wma_half_period - wma_full_period
    hma_series = diff_wma.rolling(window=sqrt_length).apply(lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    return hma_series
def rsi_pine(prices_ohlc4, period=10):
    deltas = prices_ohlc4.diff(); gains = deltas.where(deltas > 0, 0.0); losses = -deltas.where(deltas < 0, 0.0)
    avg_gains = rma(gains, period); avg_losses = rma(losses, period)
    rs = avg_gains / avg_losses.replace(0, 1e-9); rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)
def adx_pine(high, low, close, period=14):
    tr1 = high - low; tr2 = abs(high - close.shift(1)); tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1); atr = rma(tr, period)
    up_move = high.diff(); down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    safe_atr = atr.replace(0, 1e-9)
    plus_di = 100 * (rma(plus_dm, period) / safe_atr); minus_di = 100 * (rma(minus_dm, period) / safe_atr)
    dx_denominator = (plus_di + minus_di).replace(0, 1e-9); dx = 100 * (abs(plus_di - minus_di) / dx_denominator)
    adx_series = rma(dx, period)
    return adx_series.fillna(0)
def heiken_ashi_pine(df_ohlc):
    ha_df = pd.DataFrame(index=df_ohlc.index)
    if df_ohlc.empty: ha_df['HA_Open'] = pd.Series(dtype=float); ha_df['HA_Close'] = pd.Series(dtype=float); return ha_df['HA_Open'], ha_df['HA_Close']
    ha_df['HA_Close'] = (df_ohlc['Open'] + df_ohlc['High'] + df_ohlc['Low'] + df_ohlc['Close']) / 4; ha_df['HA_Open'] = np.nan
    if not df_ohlc.empty:
        ha_df.iloc[0, ha_df.columns.get_loc('HA_Open')] = (df_ohlc['Open'].iloc[0] + df_ohlc['Close'].iloc[0]) / 2
        for i in range(1, len(df_ohlc)): ha_df.iloc[i, ha_df.columns.get_loc('HA_Open')] = (ha_df.iloc[i-1, ha_df.columns.get_loc('HA_Open')] + ha_df.iloc[i-1, ha_df.columns.get_loc('HA_Close')]) / 2
    return ha_df['HA_Open'], ha_df['HA_Close']
def smoothed_heiken_ashi_pine(df_ohlc, len1=10, len2=10):
    ema_open = ema(df_ohlc['Open'], len1); ema_high = ema(df_ohlc['High'], len1); ema_low = ema(df_ohlc['Low'], len1); ema_close = ema(df_ohlc['Close'], len1)
    ha_intermediate_df = pd.DataFrame({'Open': ema_open, 'High': ema_high, 'Low': ema_low, 'Close': ema_close}, index=df_ohlc.index)
    ha_open_intermediate, ha_close_intermediate = heiken_ashi_pine(ha_intermediate_df)
    smoothed_ha_open = ema(ha_open_intermediate, len2); smoothed_ha_close = ema(ha_close_intermediate, len2)
    return smoothed_ha_open, smoothed_ha_close
def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    # S'assurer qu'il y a assez de données pour les rolling windows
    min_len_required = max(tenkan_p, kijun_p, senkou_b_p)
    if len(df_high) < min_len_required or len(df_low) < min_len_required or len(df_close) < min_len_required:
        print(f"Ichimoku: Pas assez de données ({len(df_close)} barres) pour les périodes requises (max {min_len_required}).")
        return 0 # Signal neutre/indéterminé

    tenkan_sen = (df_high.rolling(window=tenkan_p).max() + df_low.rolling(window=tenkan_p).min()) / 2
    kijun_sen = (df_high.rolling(window=kijun_p).max() + df_low.rolling(window=kijun_p).min()) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (df_high.rolling(window=senkou_b_p).max() + df_low.rolling(window=senkou_b_p).min()) / 2
    
    # Vérifier si les dernières valeurs sont NaN (ce qui peut arriver si len(data) < period)
    if pd.isna(df_close.iloc[-1]) or pd.isna(senkou_span_a.iloc[-1]) or pd.isna(senkou_span_b.iloc[-1]):
        print("Ichimoku: Valeurs NaN détectées dans close ou spans pour la dernière bougie.")
        return 0 # Indeterminate

    current_close = df_close.iloc[-1]; current_ssa = senkou_span_a.iloc[-1]; current_ssb = senkou_span_b.iloc[-1]
    cloud_top_now = max(current_ssa, current_ssb); cloud_bottom_now = min(current_ssa, current_ssb)
    signal = 0
    if current_close > cloud_top_now: signal = 1
    elif current_close < cloud_bottom_now: signal = -1
    return signal

# --- Fonction get_data utilisant Alpaca (VERSION DE DÉBOGAGE AMÉLIORÉE) ---
@st.cache_data(ttl=300)
def get_data_alpaca(symbol_alpaca: str, timeframe_str: str = "1H", limit_bars: int = 250):
    global api 
    if api is None:
        st.error("FATAL: L'objet API Alpaca n'est pas initialisé avant l'appel à get_data_alpaca.")
        print("FATAL: L'objet API Alpaca n'est pas initialisé avant l'appel à get_data_alpaca.")
        return None
        
    print(f"get_data_alpaca appelée pour: symbol='{symbol_alpaca}', timeframe='{timeframe_str}', limit={limit_bars}")

    try:
        alpaca_timeframe_object = TIMEFRAME_MAP_ALPACA.get(timeframe_str)
        if alpaca_timeframe_object is None:
            st.error(f"Chaîne de timeframe '{timeframe_str}' non valide. Symbole: {symbol_alpaca}")
            print(f"Chaîne de timeframe '{timeframe_str}' non valide pour {symbol_alpaca}. Valeurs: {list(TIMEFRAME_MAP_ALPACA.keys())}")
            return None

        print(f"Tentative d'appel à api.get_bars pour {symbol_alpaca} avec timeframe obj: {alpaca_timeframe_object}...")
        bars_df = api.get_bars(
            symbol_alpaca,
            alpaca_timeframe_object,
            limit=limit_bars + 50 # Marge pour calculs
        ).df
        print(f"Appel api.get_bars pour {symbol_alpaca} OK. Reçu: {len(bars_df)} lignes.")
        
        if bars_df.index.tz is None: bars_df.index = bars_df.index.tz_localize('UTC')
        else: bars_df.index = bars_df.index.tz_convert('UTC')

        if bars_df.empty or len(bars_df) < 100: # Seuil de données minimum
            # Ce message apparaîtra dans l'UI de Streamlit
            st.warning(f"Données Alpaca insuffisantes ou vides pour {symbol_alpaca} ({len(bars_df)} barres). Requis: 100.")
            print(f"Données Alpaca insuffisantes ou vides pour {symbol_alpaca} ({len(bars_df)} barres). Requis: 100.")
            return None

        bars_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        processed_df = bars_df.dropna()
        print(f"Données pour {symbol_alpaca} traitées. Retour de {len(processed_df)} lignes après dropna.")
        return processed_df

    except tradeapi.rest.APIError as api_err:
        st.error(f"Erreur API Alpaca pour {symbol_alpaca} (TF: {timeframe_str}): {api_err}")
        print(f"ERREUR API Alpaca pour {symbol_alpaca} (TF: {timeframe_str}):")
        print(str(api_err)) # Afficher l'erreur API complète dans les logs
        if "not found for symbol" in str(api_err) or "does not exist" in str(api_err) or "not a tradable asset" in str(api_err):
            st.warning(f"Symbole {symbol_alpaca} non trouvé ou non tradable sur Alpaca. Feed: {api.data_feed if api else 'N/A'}.")
        elif "forbidden" in str(api_err) or "subscription" in str(api_err):
             st.error(f"Accès interdit ou problème de souscription aux données pour {symbol_alpaca} sur Alpaca.")
        return None
        
    except Exception as e:
        st.error(f"Erreur inattendue dans get_data_alpaca pour {symbol_alpaca} (TF: {timeframe_str}).")
        st.exception(e) 
        print(f"ERREUR INATTENDUE dans get_data_alpaca pour {symbol_alpaca} (TF: {timeframe_str}):")
        print(traceback.format_exc())
        return None

# --- Fonctions calculate_all_signals_pine et get_stars_pine (INCHANGÉES) ---
def calculate_all_signals_pine(data):
    if data is None or len(data) < 60: print(f"calculate_all_signals: Données non fournies ou trop courtes ({len(data) if data is not None else 'None'} lignes)."); return None
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols): print(f"calculate_all_signals: Colonnes OHLC manquantes."); return None
    close = data['Close']; high = data['High']; low = data['Low']; open_price = data['Open']
    ohlc4 = (open_price + high + low + close) / 4
    bull_confluences = 0; bear_confluences = 0; signal_details_pine = {}
    try: # HMA
        hma_series = hull_ma_pine(close, 20)
        if len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            hma_val = hma_series.iloc[-1]; hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev: bull_confluences += 1; signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev: bear_confluences += 1; signal_details_pine['HMA'] = "▼"
            else: signal_details_pine['HMA'] = "─"
        else: signal_details_pine['HMA'] = "N/A"
    except Exception as e: signal_details_pine['HMA'] = f"ErrHMA"; print(f"Erreur HMA: {e}")
    try: # RSI
        rsi_series = rsi_pine(ohlc4, 10)
        if len(rsi_series) >=1 and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]; signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50: bull_confluences += 1; signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50: bear_confluences += 1; signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else: signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else: signal_details_pine['RSI'] = "N/A"
    except Exception as e: signal_details_pine['RSI'] = f"ErrRSI"; signal_details_pine['RSI_val'] = "N/A"; print(f"Erreur RSI: {e}")
    try: # ADX
        adx_series = adx_pine(high, low, close, 14)
        if len(adx_series) >= 1 and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]; signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20: bull_confluences += 1; bear_confluences += 1; signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
            else: signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else: signal_details_pine['ADX'] = "N/A"
    except Exception as e: signal_details_pine['ADX'] = f"ErrADX"; signal_details_pine['ADX_val'] = "N/A"; print(f"Erreur ADX: {e}")
    try: # HA
        ha_open, ha_close = heiken_ashi_pine(data)
        if len(ha_open) >=1 and len(ha_close) >=1 and not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['HA'] = "▼"
            else: signal_details_pine['HA'] = "─"
        else: signal_details_pine['HA'] = "N/A"
    except Exception as e: signal_details_pine['HA'] = f"ErrHA"; print(f"Erreur HA: {e}")
    try: # SHA
        sha_open, sha_close = smoothed_heiken_ashi_pine(data, len1=10, len2=10)
        if len(sha_open) >=1 and len(sha_close) >=1 and not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['SHA'] = "▼"
            else: signal_details_pine['SHA'] = "─"
        else: signal_details_pine['SHA'] = "N/A"
    except Exception as e: signal_details_pine['SHA'] = f"ErrSHA"; print(f"Erreur SHA: {e}")
    try: # Ichi
        ichi_signal = ichimoku_pine_signal(high, low, close) # La vérification de longueur est dans la fonction ichi
        if ichi_signal == 1: bull_confluences += 1; signal_details_pine['Ichi'] = "▲"
        elif ichi_signal == -1: bear_confluences += 1; signal_details_pine['Ichi'] = "▼"
        elif ichi_signal == 0 and len(data) < max(9,26,52): signal_details_pine['Ichi'] = "N/D" # Not enough data
        else: signal_details_pine['Ichi'] = "─"
    except Exception as e: signal_details_pine['Ichi'] = f"ErrIchi"; print(f"Erreur Ichi: {e}")
    
    confluence_value = max(bull_confluences, bear_confluences)
    direction = "NEUTRE"
    if bull_confluences > bear_confluences: direction = "HAUSSIER"
    elif bear_confluences > bull_confluences: direction = "BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0: direction = "CONFLIT"
    return {'confluence_P': confluence_value, 'direction_P': direction, 'bull_P': bull_confluences, 'bear_P': bear_confluences,
            'rsi_P': signal_details_pine.get('RSI_val', "N/A"), 'adx_P': signal_details_pine.get('ADX_val', "N/A"),
            'signals_P': signal_details_pine}

def get_stars_pine(confluence_value):
    if confluence_value == 6: return "⭐⭐⭐⭐⭐⭐"
    elif confluence_value == 5: return "⭐⭐⭐⭐⭐"
    elif confluence_value == 4: return "⭐⭐⭐⭐"
    elif confluence_value == 3: return "⭐⭐⭐"
    elif confluence_value == 2: return "⭐⭐"
    elif confluence_value == 1: return "⭐"
    else: return "WAIT"

# --- Interface Utilisateur ---
col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("⚙️ Paramètres")
    min_confluence_filter = st.selectbox("Confluence minimum (0-6)", options=[0, 1, 2, 3, 4, 5, 6], index=3, format_func=lambda x: f"{x} (confluence)")
    show_all_pairs = st.checkbox("Voir toutes les paires (ignorer filtre confluence)")
    # S'assurer que 'api' est disponible avant d'activer le bouton
    scan_button_disabled = api is None 
    scan_button_tooltip = "Connexion à Alpaca échouée. Vérifiez les secrets et la console." if scan_button_disabled else "Lancer le scan avec les données Alpaca"
    
    scan_button = st.button("🔍 Scanner (Données Alpaca H1)", type="primary", use_container_width=True, 
                            disabled=scan_button_disabled, help=scan_button_tooltip)

with col2:
    if scan_button: # Implique que api n'est pas None
        st.info(f"🔄 Scan en cours avec les données Alpaca (H1)...")
        processed_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, symbol_to_scan in enumerate(FOREX_PAIRS_ALPACA):
            current_progress = (i + 1) / len(FOREX_PAIRS_ALPACA)
            progress_bar.progress(current_progress)
            pair_name_display = symbol_to_scan 
            status_text.text(f"Analyse (Alpaca H1): {pair_name_display} ({i+1}/{len(FOREX_PAIRS_ALPACA)})")
            
            data_h1_alpaca = get_data_alpaca(symbol_to_scan, timeframe_str="1H", limit_bars=250)
            
            # ---- DÉBOGAGE OHLC (Optionnel, décommente si besoin pour une paire spécifique) ----
            # if data_h1_alpaca is not None and (symbol_to_scan == 'GBPUSD'): 
            #     st.write(f"--- Données OHLC pour {pair_name_display} (5 dernières barres d'Alpaca) ---")
            #     st.dataframe(data_h1_alpaca[['Open', 'High', 'Low', 'Close']].tail(5))
            # ---- FIN DÉBOGAGE ----

            if data_h1_alpaca is not None:
                signals = calculate_all_signals_pine(data_h1_alpaca)
                if signals:
                    stars_str = get_stars_pine(signals['confluence_P'])
                    result_data = {'Paire': pair_name_display, 'Direction': signals['direction_P'],
                                   'Conf. (0-6)': signals['confluence_P'], 'Étoiles': stars_str,
                                   'RSI': signals['rsi_P'], 'ADX': signals['adx_P'],
                                   'Bull': signals['bull_P'], 'Bear': signals['bear_P'],
                                   'details': signals['signals_P']}
                    processed_results.append(result_data)
                else: 
                    processed_results.append({'Paire': pair_name_display, 'Direction': 'ERREUR CALCUL', 'Conf. (0-6)':0, 
                                              'Étoiles':'N/A', 'RSI':'N/A', 'ADX':'N/A', 'Bull':0, 'Bear':0, 
                                              'details':{'Info': 'Calcul des signaux (Alpaca) a échoué'}})
            else: 
                processed_results.append({'Paire': pair_name_display, 'Direction': 'ERREUR DONNÉES', 'Conf. (0-6)':0, 
                                          'Étoiles':'N/A', 'RSI':'N/A', 'ADX':'N/A', 'Bull':0, 'Bear':0, 
                                          'details':{'Info': 'Données Alpaca non disponibles ou symbole invalide (vérifiez les logs serveur et messages UI)'}})
            time.sleep(0.25) # Léger délai pour l'API

        progress_bar.empty(); status_text.empty()

        if processed_results:
            df_all = pd.DataFrame(processed_results)
            if not show_all_pairs:
                df_display = df_all[df_all['Conf. (0-6)'] >= min_confluence_filter].copy()
                st.success(f"🎯 {len(df_display)} paire(s) avec {min_confluence_filter}+ de confluence trouvée(s) (Données Alpaca).")
            else:
                df_display = df_all.copy()
                st.info(f"🔍 Affichage des {len(df_display)} paires analysées (Données Alpaca).")
            
            if not df_display.empty:
                df_display_sorted = df_display.sort_values('Conf. (0-6)', ascending=False)
                cols_to_show_in_df = ['Paire', 'Direction', 'Conf. (0-6)', 'Étoiles', 'RSI', 'ADX', 'Bull', 'Bear']
                # Filtrer les colonnes qui existent réellement dans le DataFrame pour éviter les erreurs
                valid_cols_to_show = [col for col in cols_to_show_in_df if col in df_display_sorted.columns]
                st.dataframe(df_display_sorted[valid_cols_to_show], use_container_width=True, hide_index=True)
                
                with st.expander("📊 Détails des signaux (Données Alpaca)"):
                    for _, row in df_display_sorted.iterrows():
                        # S'assurer que 'details' existe et est un dictionnaire
                        sig_map_details = row.get('details', {})
                        if not isinstance(sig_map_details, dict): sig_map_details = {'Info': 'Détails non disponibles'}

                        st.write(f"**{row.get('Paire','N/A')}** - {row.get('Étoiles','N/A')} (Conf: {row.get('Conf. (0-6)','N/A')}) - Dir: {row.get('Direction','N/A')}")
                        detail_cols = st.columns(6)
                        pine_signals_order_for_details = ['HMA', 'RSI', 'ADX', 'HA', 'SHA', 'Ichi']
                        for idx, sig_key in enumerate(pine_signals_order_for_details):
                            detail_cols[idx].metric(label=sig_key, value=sig_map_details.get(sig_key, "N/P"))
                        st.divider()
            else: 
                st.warning(f"❌ Aucune paire ne correspond à vos critères de filtrage (min confluence: {min_confluence_filter}).")
                if show_all_pairs and df_all.empty:
                    st.error("Aucun résultat n'a été traité. Vérifiez les logs pour des erreurs de connexion ou de données initiales.")
        else: 
            st.error("❌ Aucune paire n'a pu être traitée. Vérifiez les logs de la console Streamlit Cloud pour des erreurs détaillées.")

# --- Section d'information ---
with st.expander("ℹ️ Comment ça marche (Logique Pine Script avec Données Alpaca)"):
    st.markdown("""
    **6 Signaux de Confluence analysés (inspiré par 'Canadian Confluence Premium'):**
    1.  **HMA (20)**, 2.  **RSI (10) ohlc4**, 3.  **ADX (14)** force >= 20,
    4.  **Heiken Ashi (Simple)**, 5.  **Smoothed Heiken Ashi (10,10)**, 6.  **Ichimoku Cloud (9,26,52)**.
    **Comptage & Étoiles (Logique Pine):** Confluence Finale = max(bull, bear). Étoiles basées sur score 0-6.
    **Source des Données:** Marchés Alpaca (via API).
    """)
st.caption("Scanner H1 (Données Alpaca). Les filtres Multi-TF (D1/H4) ne sont pas actifs dans cette version.")
