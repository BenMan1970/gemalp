import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback # Pour le traceback détaillé dans les logs

# Import pour Alpaca
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame 

# --- Configuration de la page et titre ---
st.set_page_config(page_title="Scanner Confluence Forex (Alpaca)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Alpaca)")
st.markdown("*Utilisation de l'API Alpaca pour les données de marché*")

# --- Récupération sécurisée des Clés API Alpaca depuis Streamlit Secrets ---
api = None 
try:
    API_KEY = st.secrets["ALPACA_API_KEY"]
    API_SECRET = st.secrets["ALPACA_SECRET_KEY"]
    BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets") 
except KeyError as e:
    st.error(f"Erreur: La clé secrète Streamlit '{e.args[0]}' n'est pas définie. Config secrets Alpaca.")
    st.stop()
except Exception as e: 
    st.error(f"Erreur lecture des secrets Streamlit: {e}")
    st.stop()

try:
    api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL)
    account = api.get_account()
    st.sidebar.success(f"Connecté compte Alpaca (Paper): {account.account_number}")
except Exception as e:
    st.error(f"Erreur initialisation/connexion API Alpaca: {e}")
    st.sidebar.error("Échec connexion Alpaca.")
    api = None 
    
# --- Liste des paires Forex (Format Alpaca) ---
FOREX_PAIRS_ALPACA = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
    'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY',
    'GBPJPY', 'EURGBP'
]

# --- Mapping pour les Timeframes Alpaca ---
TIMEFRAME_MAP_ALPACA = {
    "1Min": TimeFrame.Minute, "5Min": TimeFrame(5, tradeapi.rest.TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, tradeapi.rest.TimeFrameUnit.Minute), "1H": TimeFrame.Hour,
    "4H": TimeFrame(4, tradeapi.rest.TimeFrameUnit.Hour), "1D": TimeFrame.Day
}

# --- Fonctions d'indicateurs techniques (INCHANGÉES) ---
def ema(series, period): return series.ewm(span=period, adjust=False).mean()
def rma(series, period): return series.ewm(alpha=1/period, adjust=False).mean()
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
    rs = avg_gains / avg_losses.replace(0, 1e-9); rsi = 100 - (100 / (1 + rs)); return rsi.fillna(50)
def adx_pine(high, low, close, period=14):
    tr1 = high - low; tr2 = abs(high - close.shift(1)); tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1); atr = rma(tr, period)
    up_move = high.diff(); down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    safe_atr = atr.replace(0, 1e-9)
    plus_di = 100 * (rma(plus_dm, period) / safe_atr); minus_di = 100 * (rma(minus_dm, period) / safe_atr)
    dx_denominator = (plus_di + minus_di).replace(0, 1e-9); dx = 100 * (abs(plus_di - minus_di) / dx_denominator)
    adx_series = rma(dx, period); return adx_series.fillna(0)
def heiken_ashi_pine(df_ohlc):
    ha_df = pd.DataFrame(index=df_ohlc.index)
    if df_ohlc.empty: ha_df['HA_Open']=pd.Series(dtype=float); ha_df['HA_Close']=pd.Series(dtype=float); return ha_df['HA_Open'], ha_df['HA_Close']
    ha_df['HA_Close']=(df_ohlc['Open']+df_ohlc['High']+df_ohlc['Low']+df_ohlc['Close'])/4; ha_df['HA_Open']=np.nan
    if not df_ohlc.empty:
        ha_df.iloc[0, ha_df.columns.get_loc('HA_Open')]=(df_ohlc['Open'].iloc[0]+df_ohlc['Close'].iloc[0])/2
        for i in range(1, len(df_ohlc)): ha_df.iloc[i, ha_df.columns.get_loc('HA_Open')]=(ha_df.iloc[i-1, ha_df.columns.get_loc('HA_Open')]+ha_df.iloc[i-1, ha_df.columns.get_loc('HA_Close')])/2
    return ha_df['HA_Open'], ha_df['HA_Close']
def smoothed_heiken_ashi_pine(df_ohlc, len1=10, len2=10):
    ema_open=ema(df_ohlc['Open'],len1); ema_high=ema(df_ohlc['High'],len1); ema_low=ema(df_ohlc['Low'],len1); ema_close=ema(df_ohlc['Close'],len1)
    ha_intermediate_df=pd.DataFrame({'Open':ema_open,'High':ema_high,'Low':ema_low,'Close':ema_close},index=df_ohlc.index)
    ha_open_intermediate,ha_close_intermediate=heiken_ashi_pine(ha_intermediate_df)
    smoothed_ha_open=ema(ha_open_intermediate,len2); smoothed_ha_close=ema(ha_close_intermediate,len2)
    return smoothed_ha_open, smoothed_ha_close
def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    min_len_req=max(tenkan_p,kijun_p,senkou_b_p)
    if len(df_high)<min_len_req or len(df_low)<min_len_req or len(df_close)<min_len_req: print(f"Ichimoku: Données insuffisantes ({len(df_close)} barres) vs requis {min_len_req}."); return 0
    tenkan_sen=(df_high.rolling(window=tenkan_p).max()+df_low.rolling(window=tenkan_p).min())/2
    kijun_sen=(df_high.rolling(window=kijun_p).max()+df_low.rolling(window=kijun_p).min())/2
    senkou_span_a=(tenkan_sen+kijun_sen)/2; senkou_span_b=(df_high.rolling(window=senkou_b_p).max()+df_low.rolling(window=senkou_b_p).min())/2
    if pd.isna(df_close.iloc[-1]) or pd.isna(senkou_span_a.iloc[-1]) or pd.isna(senkou_span_b.iloc[-1]): print("Ichimoku: NaN close/spans."); return 0
    current_close=df_close.iloc[-1]; current_ssa=senkou_span_a.iloc[-1]; current_ssb=senkou_span_b.iloc[-1]
    cloud_top_now=max(current_ssa,current_ssb); cloud_bottom_now=min(current_ssa,current_ssb)
    signal=0
    if current_close>cloud_top_now: signal=1
    elif current_close<cloud_bottom_now: signal=-1
    return signal

# --- Fonction get_data utilisant Alpaca (VERSION DE DÉBOGAGE DÉTAILLÉ) ---
@st.cache_data(ttl=300)
def get_data_alpaca(symbol_alpaca: str, timeframe_str: str = "1H", limit_bars: int = 250):
    global api 
    if api is None:
        st.error("FATAL: API Alpaca non initialisée (get_data_alpaca).")
        print("FATAL: API Alpaca non initialisée (get_data_alpaca).")
        return None
    print(f"\n--- Début get_data_alpaca pour: symbol='{symbol_alpaca}', timeframe='{timeframe_str}', limit={limit_bars} ---")
    try:
        alpaca_timeframe_object = TIMEFRAME_MAP_ALPACA.get(timeframe_str)
        if alpaca_timeframe_object is None:
            st.error(f"TF '{timeframe_str}' non valide. Sym: {symbol_alpaca}. Valides: {list(TIMEFRAME_MAP_ALPACA.keys())}")
            print(f"TF '{timeframe_str}' non valide. Sym: {symbol_alpaca}. Valides: {list(TIMEFRAME_MAP_ALPACA.keys())}")
            return None
        print(f"Appel api.get_bars: sym={symbol_alpaca}, tf_obj={alpaca_timeframe_object}, limit={limit_bars + 50}")
        bars_list_raw = api.get_bars(symbol_alpaca, alpaca_timeframe_object, limit=limit_bars + 50)
        print(f"--- Inspection données brutes pour {symbol_alpaca} ---")
        print(f"Type de bars_list_raw: {type(bars_list_raw)}")
        raw_data_len = 0
        if hasattr(bars_list_raw, '__len__'): raw_data_len = len(bars_list_raw) # Pour BarSet, len donne le nombre de symboles
        # Si bars_list_raw est un BarSet, il faut accéder aux barres via la clé du symbole
        actual_bars_for_symbol = []
        if isinstance(bars_list_raw, tradeapi.rest.BarSet): # ou type(bars_list_raw).__name__ == 'BarSet'
            if symbol_alpaca in bars_list_raw:
                actual_bars_for_symbol = bars_list_raw[symbol_alpaca]
                print(f"Nombre d'objets Bar réels pour {symbol_alpaca} dans BarSet: {len(actual_bars_for_symbol)}")
                if len(actual_bars_for_symbol) > 0:
                    print(f"Premier objet Bar réel: {actual_bars_for_symbol[0]}")
                    print(f"Attributs du premier Bar réel: {vars(actual_bars_for_symbol[0])}")
            else:
                print(f"Symbole {symbol_alpaca} non trouvé dans le BarSet retourné.")
        else: # Si ce n'est pas un BarSet, peut-être une liste directe (moins probable pour get_bars)
            actual_bars_for_symbol = bars_list_raw 
            if hasattr(actual_bars_for_symbol, '__len__') and len(actual_bars_for_symbol) > 0:
                 print(f"Premier objet Bar (liste directe): {actual_bars_for_symbol[0]}")
                 print(f"Attributs du premier Bar (liste directe): {vars(actual_bars_for_symbol[0])}")
            else:
                 print(f"bars_list_raw n'est pas un BarSet et la liste est vide ou n'a pas de longueur.")

        bars_df = bars_list_raw.df
        print(f"DataFrame bars_df créé. Index type: {type(bars_df.index)}, Colonnes: {bars_df.columns.tolist()}")
        if not bars_df.empty: print(f"Head de bars_df:\n{bars_df.head()}")
        else: print(f"bars_df est vide après .df pour {symbol_alpaca}")
        print(f"--- Fin inspection données brutes {symbol_alpaca} ---")
        
        if bars_df.empty:
            # Ce message apparaîtra si .df est vide, même si bars_list_raw contenait des barres pour d'autres symboles (non applicable ici)
            # Ou si actual_bars_for_symbol était vide.
            st.warning(f"Données Alpaca vides pour {symbol_alpaca} après conversion en DataFrame.")
            print(f"Données Alpaca vides pour {symbol_alpaca} après .df.")
            return None

        if isinstance(bars_df.index, pd.DatetimeIndex):
            if bars_df.index.tz is None: bars_df.index = bars_df.index.tz_localize('UTC'); print(f"Index pour {symbol_alpaca} localisé UTC.")
            else: bars_df.index = bars_df.index.tz_convert('UTC'); print(f"Index pour {symbol_alpaca} converti UTC.")
        else:
            st.error(f"Index DF pour {symbol_alpaca} non DatetimeIndex. Type: {type(bars_df.index)}")
            print(f"Index DF pour {symbol_alpaca} non DatetimeIndex. Type: {type(bars_df.index)}")
            return None 

        if len(bars_df) < 100: 
            st.warning(f"Données Alpaca insuffisantes pour {symbol_alpaca} ({len(bars_df)} barres). Requis: 100.")
            print(f"Données Alpaca insuffisantes pour {symbol_alpaca} ({len(bars_df)} barres). Requis: 100.")
            return None
        bars_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        processed_df = bars_df.dropna()
        print(f"Données pour {symbol_alpaca} OK. Retour de {len(processed_df)} lignes après dropna.\n--- Fin get_data_alpaca pour {symbol_alpaca} ---\n")
        return processed_df
    except tradeapi.rest.APIError as api_err:
        st.error(f"Erreur API Alpaca pour {symbol_alpaca} (TF: {timeframe_str}): {api_err}")
        print(f"ERREUR API ALPACA pour {symbol_alpaca} (TF: {timeframe_str}):\n{str(api_err)}\n--- Fin get_data_alpaca pour {symbol_alpaca} (APIError) ---\n")
        if "not found for symbol" in str(api_err) or "does not exist" in str(api_err) or "not a tradable asset" in str(api_err):
            st.warning(f"Symbole {symbol_alpaca} non trouvé/non tradable sur Alpaca. Feed: {api.data_feed if api else 'N/A'}.")
        elif "forbidden" in str(api_err) or "subscription" in str(api_err):
             st.error(f"Accès interdit/souscription pour {symbol_alpaca} sur Alpaca.")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue get_data_alpaca pour {symbol_alpaca} (TF: {timeframe_str}).")
        st.exception(e) 
        print(f"ERREUR INATTENDUE get_data_alpaca pour {symbol_alpaca} (TF: {timeframe_str}):\n{traceback.format_exc()}\n--- Fin get_data_alpaca pour {symbol_alpaca} (Exception) ---\n")
        return None

# --- Fonctions calculate_all_signals_pine (INCHANGÉE) ---
def calculate_all_signals_pine(data):
    if data is None or len(data) < 60: print(f"calc_signals: Données None/courtes ({len(data) if data is not None else 'None'})."); return None
    req_cols=['Open','High','Low','Close']; 
    if not all(c in data.columns for c in req_cols): print("calc_signals: Cols OHLC manquantes."); return None
    close=data['Close']; high=data['High']; low=data['Low']; open_p=data['Open']; ohlc4=(open_p+high+low+close)/4
    bull_c=0; bear_c=0; sig_det={}
    try:
        hma_s=hull_ma_pine(close,20)
        if len(hma_s)>=2 and not hma_s.iloc[-2:].isna().any():
            hma_v=hma_s.iloc[-1];hma_p=hma_s.iloc[-2]
            if hma_v>hma_p: bull_c+=1;sig_det['HMA']="▲"
            elif hma_v<hma_p: bear_c+=1;sig_det['HMA']="▼"
            else: sig_det['HMA']="─"
        else: sig_det['HMA']="N/A"
    except Exception as e: sig_det['HMA']=f"ErrHMA"; print(f"Err HMA:{e}")
    try:
        rsi_s=rsi_pine(ohlc4,10)
        if len(rsi_s)>=1 and not pd.isna(rsi_s.iloc[-1]):
            rsi_v=rsi_s.iloc[-1];sig_det['RSI_val']=f"{rsi_v:.0f}"
            if rsi_v>50: bull_c+=1;sig_det['RSI']=f"▲({rsi_v:.0f})"
            elif rsi_v<50: bear_c+=1;sig_det['RSI']=f"▼({rsi_v:.0f})"
            else: sig_det['RSI']=f"─({rsi_v:.0f})"
        else: sig_det['RSI']="N/A"
    except Exception as e: sig_det['RSI']=f"ErrRSI";sig_det['RSI_val']="N/A";print(f"Err RSI:{e}")
    try:
        adx_s=adx_pine(high,low,close,14)
        if len(adx_s)>=1 and not pd.isna(adx_s.iloc[-1]):
            adx_v=adx_s.iloc[-1];sig_det['ADX_val']=f"{adx_v:.0f}"
            if adx_v>=20: bull_c+=1;bear_c+=1;sig_det['ADX']=f"✔({adx_v:.0f})"
            else: sig_det['ADX']=f"✖({adx_v:.0f})"
        else: sig_det['ADX']="N/A"
    except Exception as e: sig_det['ADX']=f"ErrADX";sig_det['ADX_val']="N/A";print(f"Err ADX:{e}")
    try:
        ha_o,ha_c=heiken_ashi_pine(data)
        if len(ha_o)>=1 and len(ha_c)>=1 and not pd.isna(ha_o.iloc[-1]) and not pd.isna(ha_c.iloc[-1]):
            if ha_c.iloc[-1]>ha_o.iloc[-1]: bull_c+=1;sig_det['HA']="▲"
            elif ha_c.iloc[-1]<ha_o.iloc[-1]: bear_c+=1;sig_det['HA']="▼"
            else: sig_det['HA']="─"
        else: sig_det['HA']="N/A"
    except Exception as e: sig_det['HA']=f"ErrHA";print(f"Err HA:{e}")
    try:
        sha_o,sha_c=smoothed_heiken_ashi_pine(data,10,10)
        if len(sha_o)>=1 and len(sha_c)>=1 and not pd.isna(sha_o.iloc[-1]) and not pd.isna(sha_c.iloc[-1]):
            if sha_c.iloc[-1]>sha_o.iloc[-1]: bull_c+=1;sig_det['SHA']="▲"
            elif sha_c.iloc[-1]<sha_o.iloc[-1]: bear_c+=1;sig_det['SHA']="▼"
            else: sig_det['SHA']="─"
        else: sig_det['SHA']="N/A"
    except Exception as e: sig_det['SHA']=f"ErrSHA";print(f"Err SHA:{e}")
    try:
        ichi_s=ichimoku_pine_signal(high,low,close)
        if ichi_s==1: bull_c+=1;sig_det['Ichi']="▲"
        elif ichi_s==-1: bear_c+=1;sig_det['Ichi']="▼"
        elif ichi_s==0 and (len(data) < max(9,26,52) or pd.isna(data['Close'].iloc[-1])): sig_det['Ichi']="N/D"
        else: sig_det['Ichi']="─"
    except Exception as e: sig_det['Ichi']=f"ErrIchi";print(f"Err Ichi:{e}")
    conf_v=max(bull_c,bear_c);direction="NEUTRE"
    if bull_c>bear_c: direction="HAUSSIER"
    elif bear_c>bull_c: direction="BAISSIER"
    elif bull_c==bear_c and bull_c>0: direction="CONFLIT"
    return {'confluence_P':conf_v,'direction_P':direction,'bull_P':bull_c,'bear_P':bear_c,
            'rsi_P':sig_det.get('RSI_val',"N/A"),'adx_P':sig_det.get('ADX_val',"N/A"),
            'signals_P':sig_det}

# --- Fonction get_stars_pine (CORRIGÉE) ---
def get_stars_pine(confluence_value):
    if confluence_value == 6: return "⭐⭐⭐⭐⭐⭐"
    elif confluence_value == 5: return "⭐⭐⭐⭐⭐"
    elif confluence_value == 4: return "⭐⭐⭐⭐"
    elif confluence_value == 3: return "⭐⭐⭐"
    elif confluence_value == 2: return "⭐⭐"
    elif confluence_value == 1: return "⭐"
    else: return "WAIT"

# --- Interface Utilisateur (INCHANGÉE) ---
col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("⚙️ Paramètres")
    min_confluence_filter = st.selectbox("Confluence minimum (0-6)", options=[0,1,2,3,4,5,6], index=3, format_func=lambda x: f"{x} (confluence)")
    show_all_pairs = st.checkbox("Voir toutes les paires (ignorer filtre confluence)")
    scan_button_disabled = api is None 
    scan_button_tooltip = "Connexion Alpaca échouée." if scan_button_disabled else "Lancer scan (Alpaca)"
    scan_button = st.button("🔍 Scanner (Données Alpaca H1)", type="primary", use_container_width=True, disabled=scan_button_disabled, help=scan_button_tooltip)

with col2:
    if scan_button:
        st.info(f"🔄 Scan en cours (Alpaca H1)...")
        processed_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, symbol_to_scan in enumerate(FOREX_PAIRS_ALPACA):
            current_progress = (i + 1) / len(FOREX_PAIRS_ALPACA)
            progress_bar.progress(current_progress)
            pair_name_display = symbol_to_scan 
            status_text.text(f"Analyse (Alpaca H1): {pair_name_display} ({i+1}/{len(FOREX_PAIRS_ALPACA)})")
            data_h1_alpaca = get_data_alpaca(symbol_to_scan, timeframe_str="1H", limit_bars=250)
            if data_h1_alpaca is not None:
                signals = calculate_all_signals_pine(data_h1_alpaca)
                if signals:
                    stars_str = get_stars_pine(signals['confluence_P'])
                    result_data = {'Paire':pair_name_display,'Direction':signals['direction_P'],'Conf. (0-6)':signals['confluence_P'],
                                   'Étoiles':stars_str,'RSI':signals['rsi_P'],'ADX':signals['adx_P'],'Bull':signals['bull_P'],
                                   'Bear':signals['bear_P'],'details':signals['signals_P']}
                    processed_results.append(result_data)
                else: processed_results.append({'Paire':pair_name_display,'Direction':'ERREUR CALCUL','Conf. (0-6)':0,'Étoiles':'N/A','RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Calcul signaux (Alpaca) échoué'}})
            else: processed_results.append({'Paire':pair_name_display,'Direction':'ERREUR DONNÉES','Conf. (0-6)':0,'Étoiles':'N/A','RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Données Alpaca non dispo/symb invalide (logs serveur)'}})
            time.sleep(0.3) 
        progress_bar.empty(); status_text.empty()
        if processed_results:
            df_all = pd.DataFrame(processed_results)
            df_display = df_all[df_all['Conf. (0-6)'] >= min_confluence_filter].copy() if not show_all_pairs else df_all.copy()
            if not show_all_pairs: st.success(f"🎯 {len(df_display)} paire(s) avec {min_confluence_filter}+ confluence (Alpaca).")
            else: st.info(f"🔍 Affichage des {len(df_display)} paires (Alpaca).")
            if not df_display.empty:
                df_display_sorted = df_display.sort_values('Conf. (0-6)', ascending=False)
                valid_cols = [c for c in ['Paire','Direction','Conf. (0-6)','Étoiles','RSI','ADX','Bull','Bear'] if c in df_display_sorted.columns]
                st.dataframe(df_display_sorted[valid_cols], use_container_width=True, hide_index=True)
                with st.expander("📊 Détails des signaux (Alpaca)"):
                    for _, row in df_display_sorted.iterrows():
                        sig_map = row.get('details',{})
                        if not isinstance(sig_map,dict): sig_map = {'Info':'Détails non dispo'}
                        st.write(f"**{row.get('Paire','N/A')}** - {row.get('Étoiles','N/A')} ({row.get('Conf. (0-6)','N/A')}) - Dir: {row.get('Direction','N/A')}")
                        det_cols = st.columns(6)
                        sig_order = ['HMA','RSI','ADX','HA','SHA','Ichi']
                        for idx, sig_k in enumerate(sig_order): det_cols[idx].metric(label=sig_k, value=sig_map.get(sig_k,"N/P"))
                        st.divider()
            else: st.warning(f"❌ Aucune paire avec critères filtrage (Alpaca). Vérifiez erreurs données/symbole.")
        else: st.error("❌ Aucune paire traitée (Alpaca). Vérifiez logs serveur.")

# --- Section d'information (INCHANGÉE) ---
with st.expander("ℹ️ Comment ça marche (Logique Pine Script avec Données Alpaca)"):
    st.markdown("""**6 Signaux Confluence:** HMA(20), RSI(10), ADX(14)>=20, HA(Simple), SHA(10,10), Ichi(9,26,52). 
                **Comptage & Étoiles:** Pine. **Source:** Alpaca API.""")
st.caption("Scanner H1 (Alpaca). Multi-TF non actif.")
