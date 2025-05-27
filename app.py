import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback 

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame 

st.set_page_config(page_title="Scanner Confluence Forex (Alpaca)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Alpaca)")
st.markdown("*Utilisation de l'API Alpaca pour les données de marché*")

api = None 
try:
    API_KEY = st.secrets["ALPACA_API_KEY"]
    API_SECRET = st.secrets["ALPACA_SECRET_KEY"]
    BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets") 
except KeyError as e:
    st.error(f"Erreur: Secret '{e.args[0]}' non défini. Configurez secrets Alpaca.")
    st.stop()
except Exception as e: 
    st.error(f"Erreur lecture secrets: {e}")
    st.stop()

try:
    api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL)
    account = api.get_account()
    st.sidebar.success(f"Connecté compte Alpaca (Paper): {account.account_number}")
except Exception as e:
    st.error(f"Erreur initialisation/connexion API Alpaca: {e}")
    st.sidebar.error("Échec connexion Alpaca.")
    api = None 
    
FOREX_PAIRS_ALPACA = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP']
TIMEFRAME_MAP_ALPACA = {
    "1Min": TimeFrame.Minute, "5Min": TimeFrame(5, tradeapi.rest.TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, tradeapi.rest.TimeFrameUnit.Minute), "1H": TimeFrame.Hour,
    "4H": TimeFrame(4, tradeapi.rest.TimeFrameUnit.Hour), "1D": TimeFrame.Day
}

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()
def hull_ma_pine(dc, p=20):
    hl=int(p/2); sl=int(np.sqrt(p))
    wma1=dc.rolling(window=hl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
    wma2=dc.rolling(window=p).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
    diff=2*wma1-wma2; return diff.rolling(window=sl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
def rsi_pine(po4,p=10): d=po4.diff();g=d.where(d>0,0.0);l=-d.where(d<0,0.0);ag=rma(g,p);al=rma(l,p);rs=ag/al.replace(0,1e-9);rsi=100-(100/(1+rs));return rsi.fillna(50)
def adx_pine(h,l,c,p=14):
    tr1=h-l;tr2=abs(h-c.shift(1));tr3=abs(l-c.shift(1));tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1);atr=rma(tr,p)
    um=h.diff();dm=l.shift(1)-l
    pdm=pd.Series(np.where((um>dm)&(um>0),um,0.0),index=h.index);mdm=pd.Series(np.where((dm>um)&(dm>0),dm,0.0),index=h.index)
    satr=atr.replace(0,1e-9);pdi=100*(rma(pdm,p)/satr);mdi=100*(rma(mdm,p)/satr)
    dxden=(pdi+mdi).replace(0,1e-9);dx=100*(abs(pdi-mdi)/dxden);return rma(dx,p).fillna(0)
def heiken_ashi_pine(dfo):
    ha=pd.DataFrame(index=dfo.index)
    if dfo.empty:ha['HA_Open']=pd.Series(dtype=float);ha['HA_Close']=pd.Series(dtype=float);return ha['HA_Open'],ha['HA_Close']
    ha['HA_Close']=(dfo['Open']+dfo['High']+dfo['Low']+dfo['Close'])/4;ha['HA_Open']=np.nan
    if not dfo.empty:
        ha.iloc[0,ha.columns.get_loc('HA_Open')]=(dfo['Open'].iloc[0]+dfo['Close'].iloc[0])/2
        for i in range(1,len(dfo)):ha.iloc[i,ha.columns.get_loc('HA_Open')]=(ha.iloc[i-1,ha.columns.get_loc('HA_Open')]+ha.iloc[i-1,ha.columns.get_loc('HA_Close')])/2
    return ha['HA_Open'],ha['HA_Close']
def smoothed_heiken_ashi_pine(dfo,l1=10,l2=10):
    eo=ema(dfo['Open'],l1);eh=ema(dfo['High'],l1);el=ema(dfo['Low'],l1);ec=ema(dfo['Close'],l1)
    hai=pd.DataFrame({'Open':eo,'High':eh,'Low':el,'Close':ec},index=dfo.index)
    hao_i,hac_i=heiken_ashi_pine(hai);sho=ema(hao_i,l2);shc=ema(hac_i,l2);return sho,shc
def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    min_len_req = max(tenkan_p, kijun_p, senkou_b_p)
    if len(df_high) < min_len_req or len(df_low) < min_len_req or len(df_close) < min_len_req: print(f"Ichimoku: Données insuffisantes ({len(df_close)} barres) vs requis {min_len_req}."); return 0 
    tenkan_sen = (df_high.rolling(window=tenkan_p).max() + df_low.rolling(window=tenkan_p).min()) / 2
    kijun_sen = (df_high.rolling(window=kijun_p).max() + df_low.rolling(window=kijun_p).min()) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2; senkou_span_b = (df_high.rolling(window=senkou_b_p).max() + df_low.rolling(window=senkou_b_p).min()) / 2
    if pd.isna(df_close.iloc[-1]) or pd.isna(senkou_span_a.iloc[-1]) or pd.isna(senkou_span_b.iloc[-1]): print("Ichimoku: NaN détecté dans close ou spans."); return 0 
    current_close = df_close.iloc[-1]; current_ssa = senkou_span_a.iloc[-1]; current_ssb = senkou_span_b.iloc[-1]
    cloud_top_now = max(current_ssa, current_ssb); cloud_bottom_now = min(current_ssa, current_ssb); sig = 0 
    if current_close > cloud_top_now: sig = 1
    elif current_close < cloud_bottom_now: sig = -1
    return sig

@st.cache_data(ttl=300)
def get_data_alpaca(symbol_alpaca: str, timeframe_str: str = "1H", limit_bars: int = 250):
    global api 
    if api is None: st.error("FATAL: API Alpaca non initialisée (get_data)."); print("FATAL: API Alpaca non initialisée (get_data)."); return None
    print(f"\n--- Début get_data_alpaca: sym='{symbol_alpaca}', tf='{timeframe_str}', lim={limit_bars} ---")
    try:
        tf_obj = TIMEFRAME_MAP_ALPACA.get(timeframe_str)
        if tf_obj is None: st.error(f"TF '{timeframe_str}' non valide. Sym: {symbol_alpaca}. Valides: {list(TIMEFRAME_MAP_ALPACA.keys())}"); print(f"TF '{timeframe_str}' non valide. Sym: {symbol_alpaca}. Valides: {list(TIMEFRAME_MAP_ALPACA.keys())}"); return None
        print(f"Appel api.get_bars: sym={symbol_alpaca}, tf_obj={tf_obj}, limit={limit_bars+50}")
        bars_list_raw = api.get_bars(symbol_alpaca, tf_obj, limit=limit_bars+50)
        print(f"--- Inspection données brutes pour {symbol_alpaca} ---"); print(f"Type de bars_list_raw: {type(bars_list_raw)}")
        bars_df = None 
        try:
            bars_df = bars_list_raw.df 
            print(f"DF bars_df créé. Index type: {type(bars_df.index)}, Cols: {bars_df.columns.tolist()}")
            if not bars_df.empty: print(f"Head de bars_df:\n{bars_df.head()}")
            else: print(f"bars_df est vide après .df pour {symbol_alpaca}.")
        except KeyError as ke: st.error(f"Erreur clé (symbole?) .df {symbol_alpaca}: {ke}"); print(f"Erreur clé .df {symbol_alpaca}: {ke}"); return None
        except Exception as df_err: st.error(f"Erreur .df {symbol_alpaca}: {df_err}"); print(f"Erreur .df {symbol_alpaca}: {df_err}"); return None
        print(f"--- Fin inspection {symbol_alpaca} ---")
        if bars_df is None or bars_df.empty: st.warning(f"Données Alpaca vides pour {symbol_alpaca} (post .df)."); print(f"Données Alpaca vides pour {symbol_alpaca} (post .df)."); return None
        if isinstance(bars_df.index, pd.DatetimeIndex):
            if bars_df.index.tz is None: bars_df.index=bars_df.index.tz_localize('UTC'); print(f"Index {symbol_alpaca} localisé UTC.")
            else: bars_df.index=bars_df.index.tz_convert('UTC'); print(f"Index {symbol_alpaca} converti UTC.")
        else: st.error(f"Index DF {symbol_alpaca} non DatetimeIndex. Type: {type(bars_df.index)}"); print(f"Index DF {symbol_alpaca} non DatetimeIndex. Type: {type(bars_df.index)}"); return None 
        if len(bars_df)<100: st.warning(f"Données Alpaca<100 {symbol_alpaca} ({len(bars_df)})."); print(f"Données<100 {symbol_alpaca} ({len(bars_df)})."); return None
        bars_df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'},inplace=True)
        pdf=bars_df.dropna(); print(f"Données {symbol_alpaca} OK. Retour {len(pdf)}l.\n--- Fin get_data {symbol_alpaca} ---\n"); return pdf
    except tradeapi.rest.APIError as api_e:
        st.error(f"Erreur API Alpaca {symbol_alpaca} (TF:{timeframe_str}): {api_e}")
        print(f"ERREUR API ALPACA {symbol_alpaca} (TF:{timeframe_str}):\n{str(api_e)}\n--- Fin get_data {symbol_alpaca}(APIError)---\n")
        if "not found" in str(api_e) or "does not exist" in str(api_e) or "not a tradable" in str(api_e): st.warning(f"Symbole {symbol_alpaca} non trouvé/tradable. Feed: {api.data_feed if api else 'N/A'}.")
        elif "forbidden" in str(api_e) or "subscription" in str(api_e): st.error(f"Accès interdit/souscription {symbol_alpaca}.")
        return None
    except Exception as e: st.error(f"Erreur inattendue get_data {symbol_alpaca} (TF:{timeframe_str})."); st.exception(e);print(f"ERREUR INATTENDUE get_data {symbol_alpaca}(TF:{timeframe_str}):\n{traceback.format_exc()}\n---Fin get_data {symbol_alpaca}(Ex)---\n");return None

# --- Fonction calculate_all_signals_pine (CORRIGÉE pour indentation) ---
def calculate_all_signals_pine(data):
    if data is None or len(data) < 60:
        print(f"calculate_all_signals: Données non fournies ou trop courtes ({len(data) if data is not None else 'None'} lignes).")
        return None
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"calculate_all_signals: Colonnes OHLC manquantes.")
        return None
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
        sha_open, sha_close = smoothed_heiken_ashi_pine(data, 10, 10)
        if len(sha_open) >=1 and len(sha_close) >=1 and not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['SHA'] = "▼"
            else: signal_details_pine['SHA'] = "─"
        else: signal_details_pine['SHA'] = "N/A"
    except Exception as e: signal_details_pine['SHA'] = f"ErrSHA"; print(f"Erreur SHA: {e}")
    try: # Ichi
        ichimoku_signal_val = ichimoku_pine_signal(high, low, close)
        if ichimoku_signal_val == 1: bull_confluences += 1; signal_details_pine['Ichi'] = "▲"
        elif ichimoku_signal_val == -1: bear_confluences += 1; signal_details_pine['Ichi'] = "▼"
        elif ichimoku_signal_val == 0 and (len(data) < max(9,26,52) or (len(data) > 0 and pd.isna(data['Close'].iloc[-1]))): signal_details_pine['Ichi'] = "N/D"
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

def get_stars_pine(cfv):
    if cfv==6:return"⭐⭐⭐⭐⭐⭐";elif cfv==5:return"⭐⭐⭐⭐⭐";elif cfv==4:return"⭐⭐⭐⭐";
    elif cfv==3:return"⭐⭐⭐";elif cfv==2:return"⭐⭐";elif cfv==1:return"⭐";else:return"WAIT"

col1,col2=st.columns([1,3])
with col1:
    st.subheader("⚙️ Paramètres");min_conf=st.selectbox("Confluence min (0-6)",options=[0,1,2,3,4,5,6],index=3,format_func=lambda x:f"{x} (confluence)")
    show_all=st.checkbox("Voir toutes les paires (ignorer filtre)");scan_dis=api is None;scan_tip="Connexion Alpaca échouée." if scan_dis else "Lancer scan (Alpaca)"
    scan_btn=st.button("🔍 Scanner (Données Alpaca H1)",type="primary",use_container_width=True,disabled=scan_dis,help=scan_tip)
with col2:
    if scan_btn:
        st.info(f"🔄 Scan en cours (Alpaca H1)...");pr_res=[];pb=st.progress(0);stx=st.empty()
        for i,sym_scan in enumerate(FOREX_PAIRS_ALPACA):
            cp=(i+1)/len(FOREX_PAIRS_ALPACA);pb.progress(cp);pnd=sym_scan;stx.text(f"Analyse (Alpaca H1):{pnd}({i+1}/{len(FOREX_PAIRS_ALPACA)})")
            d_h1_alp=get_data_alpaca(sym_scan,timeframe_str="1H",limit_bars=250)
            if d_h1_alp is not None:
                sigs=calculate_all_signals_pine(d_h1_alp)
                if sigs:strs=get_stars_pine(sigs['confluence_P']);rd={'Paire':pnd,'Direction':sigs['direction_P'],'Conf. (0-6)':sigs['confluence_P'],'Étoiles':strs,'RSI':sigs['rsi_P'],'ADX':sigs['adx_P'],'Bull':sigs['bull_P'],'Bear':sigs['bear_P'],'details':sigs['signals_P']};pr_res.append(rd)
                else:pr_res.append({'Paire':pnd,'Direction':'ERREUR CALCUL','Conf. (0-6)':0,'Étoiles':'N/A','RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Calcul signaux (Alpaca) échoué'}})
            else:pr_res.append({'Paire':pnd,'Direction':'ERREUR DONNÉES','Conf. (0-6)':0,'Étoiles':'N/A','RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Données Alpaca non dispo/symb invalide(logs serveur)'}})
            time.sleep(0.3)
        pb.empty();stx.empty()
        if pr_res:
            dfa=pd.DataFrame(pr_res);dfd=dfa[dfa['Conf. (0-6)']>=min_conf].copy()if not show_all else dfa.copy()
            if not show_all:st.success(f"🎯 {len(dfd)} paire(s) avec {min_conf}+ confluence (Alpaca).")
            else:st.info(f"🔍 Affichage des {len(dfd)} paires (Alpaca).")
            if not dfd.empty:
                dfds=dfd.sort_values('Conf. (0-6)',ascending=False);vcs=[c for c in['Paire','Direction','Conf. (0-6)','Étoiles','RSI','ADX','Bull','Bear']if c in dfds.columns]
                st.dataframe(dfds[vcs],use_container_width=True,hide_index=True)
                with st.expander("📊 Détails des signaux (Alpaca)"):
                    for _,r in dfds.iterrows():
                        sm=r.get('details',{});
                        if not isinstance(sm,dict):sm={'Info':'Détails non dispo'}
                        st.write(f"**{r.get('Paire','N/A')}** - {r.get('Étoiles','N/A')} ({r.get('Conf. (0-6)','N/A')}) - Dir: {r.get('Direction','N/A')}")
                        dc=st.columns(6);so=['HMA','RSI','ADX','HA','SHA','Ichi']
                        for idx,sk in enumerate(so):dc[idx].metric(label=sk,value=sm.get(sk,"N/P"))
                        st.divider()
            else:st.warning(f"❌ Aucune paire avec critères filtrage (Alpaca). Vérifiez erreurs données/symbole.")
        else:st.error("❌ Aucune paire traitée (Alpaca). Vérifiez logs serveur.")
with st.expander("ℹ️ Comment ça marche (Logique Pine Script avec Données Alpaca)"):
    st.markdown("""**6 Signaux Confluence:** HMA(20),RSI(10),ADX(14)>=20,HA(Simple),SHA(10,10),Ichi(9,26,52).**Comptage & Étoiles:**Pine.**Source:**Alpaca API.""")
st.caption("Scanner H1 (Alpaca). Multi-TF non actif.")
