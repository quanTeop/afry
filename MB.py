# 0) Setup
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


plt.rcParams["figure.figsize"] = (9,4)

# 1) LETTURA FILE: definizione funzioni (1.1, 1.2, 1.3)

# 1.1) Funzione per leggere i file XML dei prezzi MGP e filtrare una zona

def read_mgp_xml(files_pattern, zone="NORD"):
    paths = sorted(glob.glob(files_pattern))  # prendo tutti i file che matchano il pattern
    all_rows = []

    for path in paths:
        tree = ET.parse(path)          # apro il file xml
        root = tree.getroot()          # radice dell'albero xml

        # ogni <Prezzi> è una riga con Data, Ora e valori per zona
        for prezzi in root.findall(".//Prezzi"):
            rec = {child.tag.upper(): (child.text or "").strip() for child in prezzi}

            if zone.upper() not in rec:   # se manca la zona, salto
                continue

            # converto data + ora in timestamp orario
            day = pd.to_datetime(rec["DATA"], format="%Y%m%d")
            ora = int(rec["ORA"]) - 1     # da 1-24 a 0-23
            ts = day + pd.to_timedelta(ora, unit="h")

            # prendo il prezzo e converto la virgola in punto
            raw = rec[zone.upper()]
            if raw == "":
                continue
            prezzo = float(raw.replace(",", "."))

            all_rows.append((ts, prezzo))

    # costruisco il DataFrame finale
    df = pd.DataFrame(all_rows, columns=["datetime","prezzo_mgp"]).sort_values("datetime")
    return df

# UTILIZZO LA FUNZIONE
mgp_nord = read_mgp_xml("AFRY_MB/mgp_train/*MGPPrezzi.xml", zone="NORD")

# controllo 

# print(mgp_nord.shape)
# print(mgp_nord.head())

# il data set mostra un'ora mancante in corrispondenza del cambio d'ora (4343 ore invece che 4344)

# 1.2) funzione che legga file CSV di forecast + actual demand

def read_load(path):
    
    # i) leggo il CSV con separatore ';' # separare manualmente le colonne è inutile
    
    df = pd.read_csv(path, sep=';')

    # ii) prendo solo l'inizio dell'intervallo "start - end"
    
    start = df["Time (CET/CEST)"].astype(str).str.split(pat=" - ", n=1).str[0]
    dt = pd.to_datetime(start, dayfirst=True, errors="coerce")

    # iii) conversione numerica robusta (quella semplice dava errore per il trattino) 
    
    f_col = "Day-ahead Total Load Forecast [MW] - BZN|IT-North"
    a_col = "Actual Total Load [MW] - BZN|IT-North"

    load_forecast = pd.to_numeric(df[f_col], errors="coerce")
    load_actual   = pd.to_numeric(df[a_col], errors="coerce")

    # iv) costruzione DataFrame 
    
    out = (pd.DataFrame({
        "datetime": dt,
        "load_forecast": load_forecast,
        "load_actual": load_actual
    })
    .dropna(subset=["datetime"])
    .sort_values("datetime"))

    # v) da 15' a 60' → media (potenza in MW)
    out = (out.set_index("datetime")
              .resample("h").mean()
              .reset_index())

    # vi) calcolo delta domanda 
    out["delta_domanda"] = out["load_actual"] - out["load_forecast"]
    out["month"] = out["datetime"].dt.month
    return out
   
# UTILIZZO LA FUNZIONE
path = "AFRY_MB/load/load_forecast_actual.csv"
load_2025 = read_load(path)

# divido TRAIN e TEST

train = load_2025[load_2025["month"] <= 6].copy()                     # Gen–Giu
test  = load_2025[(load_2025["month"] >= 7) & (load_2025["month"] <= 8)].copy()  # Lug–Ago

# provo 
# print(load_2025.head())
# print("TRAIN ore:", len(train), " | TEST ore:", len(test))

# 1.2) funzione che legga file CSV di forecast + actual generation

def read_res_total(path):
    
    # i) leggo CSV 
    df = pd.read_csv(path, sep=';')

    # ii) prendo solo l'inizio dell'intervallo "start - end"
    start = df["MTU (CET/CEST)"].astype(str).str.split(" - ", n=1).str[0]
    dt = pd.to_datetime(start, dayfirst=True, errors="coerce")

    # iii) "n/e" nel csv continuava a dare errore, lo trasformo direttamente in zero
    
    def to_num(col):
        return pd.to_numeric(df[col].replace("n/e", 0), errors="coerce").fillna(0)

    # iv) colonne forecast/actual e sommo + creo dataframe
    
    f_solar    = to_num("Generation - Solar [MW] Day Ahead/ Italy (IT)")
    a_solar    = to_num("Generation - Solar [MW] Current / Italy (IT)")

    f_wind_off = to_num("Generation - Wind Offshore [MW] Day Ahead/ Italy (IT)")
    a_wind_off = to_num("Generation - Wind Offshore [MW] Current / Italy (IT)")

    f_wind_on  = to_num("Generation - Wind Onshore [MW] Day Ahead/ Italy (IT)")
    a_wind_on  = to_num("Generation - Wind Onshore [MW] Current / Italy (IT)")

    generation_forecast = f_solar + f_wind_off + f_wind_on
    generation_actual   = a_solar + a_wind_off + a_wind_on

    out = (pd.DataFrame({
        "datetime": dt,
        "generation_forecast": generation_forecast,
        "generation_actual": generation_actual
    })
    .dropna(subset=["datetime"])
    .sort_values("datetime"))

    # v) da 15' a 60' → media (potenza in MW)
    out = (out.set_index("datetime")
              .resample("H").mean()
              .reset_index())

    # vi) calcolo delta RES
    out["delta_res"] = out["generation_actual"] - out["generation_forecast"]
    out["month"] = out["datetime"].dt.month
    return out

    
# UTILIZZO LA FUNZIONE 

path_res = "AFRY_MB/res/generation_forecast_actual.csv" 
res_2025 = read_res_total(path_res) 

# provo

# print(res_2025.head()) 
# print("Ore:", len(res_2025))

# ============================================
# 1.3) LETTURA CSV MB (robusta)
# ============================================
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# separa solo le virgole FUORI dalle virgolette
_SPLIT_COMMA_OUTSIDE_QUOTES = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')

def _read_mb_csv(path):
    """Lettore robusto per i CSV Terna (campi spesso quotati)."""
    df = None
    for enc in ("utf-8", "latin1"):
        try:
            tmp = pd.read_csv(path, sep=",", quotechar='"', encoding=enc, engine="python")
            if tmp.shape[1] > 1:
                df = tmp; break
        except Exception:
            pass
    if df is None or df.shape[1] == 1:
        try:
            tmp = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
            if tmp.shape[1] > 1:
                df = tmp
        except Exception:
            pass
    if df is None or df.shape[1] == 1:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip("\r\n") for ln in f if ln.strip() != ""]
        if not lines:
            raise ValueError(f"File vuoto o illeggibile: {path}")
        cols = [c.strip().strip('"') for c in _SPLIT_COMMA_OUTSIDE_QUOTES.split(lines[0])]
        rows = []
        for ln in lines[1:]:
            parts = [p.strip().strip('"') for p in _SPLIT_COMMA_OUTSIDE_QUOTES.split(ln)]
            if len(parts) < len(cols): parts += [""]*(len(cols)-len(parts))
            rows.append(parts[:len(cols)])
        df = pd.DataFrame(rows, columns=cols)
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df

def _to_num_series(s: pd.Series) -> pd.Series:
    # pulizia spazi (inclusi NBSP), migliaia/decimali IT
    s = s.astype(str)
    for bad in ['\u00A0', '\u202F', '\u2007', ' ']:
        s = s.str.replace(bad, '', regex=False)
    s = s.str.replace('n/e', '0', regex=False)
    s = s.str.replace('.',   '', regex=False)   # migliaia
    s = s.str.replace(',',   '.', regex=False)  # decimali
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

def _parse_datetime_flex(series: pd.Series) -> pd.Series:
    txt = (series.astype(str).str.strip().str.replace('\u00A0', ' ', regex=False))
    if (txt.str.contains(" - ").any()):
        txt = txt.str.split(" - ", n=1).str[0]
    txt = txt.str.split(",", n=1).str[0]  # se è entrata l'intera riga
    for f in ["%d/%m/%Y %H:%M:%S","%d/%m/%Y %H:%M","%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M"]:
        dt = pd.to_datetime(txt, format=f, errors="coerce")
        if dt.notna().any(): return dt
    return pd.to_datetime(txt, dayfirst=True, errors="coerce")

def read_mb_monthly_quarter(pattern, zone="NORD"):
    """
    Legge uno o più 'Riepilogo_Mensile_Quarto_Orario_YYYYMM.csv' e restituisce serie ORARIE:
      datetime, sbil_netto, volume_up, volume_down, volume_mb, prezzo_mb
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Nessun file trovato per pattern: {pattern}")

    pieces = []
    for p in paths:
        df = _read_mb_csv(p)
        df.columns = [c.strip() for c in df.columns]

        cand_time   = [c for c in df.columns if "DATA" in c.upper()]
        cand_zone   = [c for c in df.columns if "MACROZONA" in c.upper()]
        cand_sbil   = [c for c in df.columns if ("SBIL" in c.upper() and "MWH" in c.upper())]
        cand_sign   = [c for c in df.columns if "SEGNO" in c.upper()]
        cand_p_sbil = [c for c in df.columns if ("PREZZO" in c.upper() and ("SBIL" in c.upper() or "SBILANCIAMENTO" in c.upper()))]
        cand_p_base = [c for c in df.columns if ("PREZZO" in c.upper() and "BASE" in c.upper())]
        cand_p_inc  = [c for c in df.columns if "INCENT" in c.upper()]

        if not (cand_time and cand_zone and cand_sbil):
            raise ValueError(f"Header inatteso in {os.path.basename(p)}.\nColonne viste: {list(df.columns)}")

        time_col, zona_col, sbil_col = cand_time[0], cand_zone[0], cand_sbil[0]

        zser = df[zona_col].astype(str).str.upper().str.strip()
        dfx = df[zser.eq(zone.upper())]
        if dfx.empty: dfx = df[zser.str.contains(zone.upper(), na=False)]
        if dfx.empty: continue

        dt_q = _parse_datetime_flex(dfx[time_col])
        if dt_q.isna().all():
            examples = dfx[time_col].astype(str).head(3).tolist()
            raise ValueError(f"Parse datetime fallito su {time_col} in {os.path.basename(p)}.\nEsempi: {examples}")

        sbil_mag = _to_num_series(dfx[sbil_col]).fillna(0.0)
        if cand_sign:
            sign = dfx[cand_sign[0]].astype(str).str.strip().map({'+': 1.0, '-': -1.0}).fillna(1.0)
            sbil_q = sbil_mag * sign
        else:
            sbil_q = sbil_mag

        if cand_p_sbil:
            p_q = _to_num_series(dfx[cand_p_sbil[0]]).fillna(0.0)
        elif (cand_p_base and cand_p_inc):
            p_q = (_to_num_series(dfx[cand_p_base[0]]) + _to_num_series(dfx[cand_p_inc[0]])).fillna(0.0)
        else:
            raise ValueError(f"Colonna prezzo non trovata in {os.path.basename(p)}.")

        pieces.append(pd.DataFrame({
            "datetime_qh": dt_q,
            "sbil_qMWh": sbil_q,
            "price_q": p_q,
        }))

    if not pieces:
        raise ValueError("Nessun dato MB parsato. Verifica nomi colonne/zone.")

    qh = (pd.concat(pieces, ignore_index=True)
            .dropna(subset=["datetime_qh"])
            .sort_values("datetime_qh"))
    qh["datetime"] = qh["datetime_qh"].dt.floor("h")

    tmp = qh.copy()
    tmp["w"] = tmp["sbil_qMWh"].abs()
    tmp["weighted_price"] = tmp["price_q"] * tmp["w"]
    grp = tmp.groupby("datetime", sort=True)

    sbil_netto  = grp["sbil_qMWh"].sum()
    volume_up   = grp["sbil_qMWh"].apply(lambda x: x[x > 0].sum()).fillna(0.0)
    volume_down = grp["sbil_qMWh"].apply(lambda x: (-x[x < 0]).sum()).fillna(0.0)
    volume_mb   = volume_up + volume_down

    num_p   = grp["weighted_price"].sum()
    w_sum   = grp["w"].sum()
    mean_p  = grp["price_q"].mean()
    prezzo_mb = num_p.divide(w_sum)
    prezzo_mb = prezzo_mb.where(w_sum > 0, mean_p)

    out = (pd.DataFrame({
        "datetime": sbil_netto.index,
        "sbil_netto": sbil_netto.values,
        "volume_up": volume_up.values,
        "volume_down": volume_down.values,
        "volume_mb": volume_mb.values,
        "prezzo_mb": prezzo_mb.values
    })
    .sort_values("datetime")
    .reset_index(drop=True))
    out["month"] = out["datetime"].dt.month
    return out

# -----------------------------
# COSTRUZIONE DATASET MB
# -----------------------------
mb_train = pd.concat([
    read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202501.csv"),
    read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202502.csv"),
    read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202503.csv"),
    read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202504.csv"),
    read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202505.csv"),
    read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202506.csv"),
], ignore_index=True)

mb_test = pd.concat([
    read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202507.csv"),
    read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202508.csv"),
], ignore_index=True)

mb_all = (pd.concat([mb_train, mb_test], ignore_index=True)
            .sort_values("datetime")
            .reset_index(drop=True))

# sanity
assert (mb_all["volume_up"]>=0).all() and (mb_all["volume_down"]>=0).all()
assert (mb_all["volume_mb"]>=0).all()
assert mb_all["prezzo_mb"].notna().mean() > 0.95
print("Ore MB totali:", len(mb_all))

# ============================================
# 2) MERGE + FEATURE SET
# ============================================
feat = (mb_all
    .merge(mgp_nord, on="datetime", how="left")                              # prezzo MGP
    .merge(load_2025[["datetime","delta_domanda"]], on="datetime", how="left")
    .merge(res_2025[["datetime","delta_res"]],       on="datetime", how="left"))

# calendario
feat["hour"]  = feat["datetime"].dt.hour
feat["dow"]   = feat["datetime"].dt.dayofweek
feat["month"] = feat["datetime"].dt.month
feat["is_we"] = (feat["dow"]>=5).astype(int)

# cicliche
feat["hour_sin"] = np.sin(2*np.pi*feat["hour"]/24)
feat["hour_cos"] = np.cos(2*np.pi*feat["hour"]/24)
feat["mth_sin"]  = np.sin(2*np.pi*(feat["month"]-1)/12)
feat["mth_cos"]  = np.cos(2*np.pi*(feat["month"]-1)/12)

# feature "fisiche": delta netto e sue trasformazioni
feat["delta_net"] = feat["delta_domanda"] - feat["delta_res"]
feat["abs_net"]   = feat["delta_net"].abs()
feat["pos_net"]   = feat["delta_net"].clip(lower=0.0)
feat["neg_net"]   = (-feat["delta_net"].clip(upper=0.0))

# opzionale: rimuovi target prezzo zero inattesi
feat.loc[feat["prezzo_mb"]<=0, "prezzo_mb"] = np.nan

# ============================================
# 2bis) SPLIT + IMPUTAZIONE + SET PER TARGET
# ============================================
train_mask = feat["month"]<=6
test_mask  = (feat["month"]>=7) & (feat["month"]<=8)

base_cols = [
    "prezzo_mgp","delta_domanda","delta_res",
    "delta_net","abs_net","pos_net","neg_net",
    "hour_sin","hour_cos","mth_sin","mth_cos","is_we"
]

# imputazione lieve (mediana del train) per MGP/ΔDom/ΔRES in test se mancanti
for c in ["prezzo_mgp","delta_domanda","delta_res","delta_net","abs_net","pos_net","neg_net"]:
    med = feat.loc[train_mask, c].median()
    feat[c] = feat[c].fillna(med)

# VOLUME
colsV = base_cols + ["volume_mb"]
trV = feat.loc[train_mask, ["datetime"]+colsV].dropna(subset=["volume_mb"])
teV = feat.loc[test_mask,  ["datetime"]+colsV].dropna(subset=["volume_mb"])
XtrV, XteV = trV[base_cols].to_numpy(), teV[base_cols].to_numpy()
ytrV, yteV = trV["volume_mb"].to_numpy(), teV["volume_mb"].to_numpy()
dt_test_V  = teV["datetime"].to_numpy()

# PREZZO
colsP = base_cols + ["prezzo_mb"]
trP = feat.loc[train_mask, ["datetime"]+colsP].dropna(subset=["prezzo_mb"])
teP = feat.loc[test_mask,  ["datetime"]+colsP].dropna(subset=["prezzo_mb"])
XtrP, XteP = trP[base_cols].to_numpy(), teP[base_cols].to_numpy()
ytrP, yteP = trP["prezzo_mb"].to_numpy(), teP["prezzo_mb"].to_numpy()
dt_test_P  = teP["datetime"].to_numpy()
# pesi per valutare il prezzo: volumi orari (business-relevance)
wP = feat.loc[test_mask & feat["prezzo_mb"].notna(), "volume_mb"].to_numpy()

print("Shapes → Volume", XtrV.shape, XteV.shape, " | Prezzo", XtrP.shape, XteP.shape)

# ============================================
# 3) MODELLI + METRICHE
# ============================================
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

def smape(y_true, y_pred, eps=1e-6):
    return (200.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)))

def eval_reg(y_true, y_pred, label="", weights=None):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    sm   = smape(y_true, y_pred)
    if weights is not None:
        wmae = np.average(np.abs(y_true - y_pred), weights=weights)
        print(f"[{label}] RMSE={rmse:,.1f}  MAE={mae:,.1f}  sMAPE={sm:,.1f}%  WMAE={wmae:,.1f}")
        return rmse, mae, sm, wmae
    else:
        print(f"[{label}] RMSE={rmse:,.1f}  MAE={mae:,.1f}  sMAPE={sm:,.1f}%")
        return rmse, mae, sm

# Volume: lineare + log1p
linV      = LinearRegression().fit(XtrV, ytrV)
predV_lin = linV.predict(XteV)
eval_reg(yteV, predV_lin, "Volume (lin)")

linV_log  = LinearRegression().fit(XtrV, np.log1p(ytrV))
predV     = np.expm1(linV_log.predict(XteV))
eval_reg(yteV, predV, "Volume (log1p)")

# Prezzo: Ridge con feature arricchite
ridgeP = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0]).fit(XtrP, ytrP)
predP  = ridgeP.predict(XteP)
eval_reg(yteP, predP, "Prezzo (ridge)", weights=wP)

# ============================================
# 4) GRAFICI
# ============================================
def plot_ts(dt, y_true, y_pred, title, ylabel):
    plt.figure(figsize=(10,3.5))
    plt.plot(dt, y_true, label="Real")
    plt.plot(dt, y_pred, label="Pred", alpha=0.9)
    plt.title(title); plt.ylabel(ylabel); plt.xlabel("")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(4.2,4.2))
    plt.scatter(y_true, y_pred, s=6)
    lim = [0, max(float(np.nanmax(y_true)), float(np.nanmax(y_pred)))]
    plt.plot(lim, lim, lw=1)
    plt.title(title); plt.xlabel("Real"); plt.ylabel("Pred")
    plt.tight_layout(); plt.show()

plot_ts(dt_test_V, yteV, predV, "Volume MB – Test (real vs pred)", "MWh")
plot_scatter(yteV, predV, "Volume MB – Pred vs Real (test)")
plot_ts(dt_test_P, yteP, predP, "Prezzo MB – Test (real vs pred)", "€/MWh")
plot_scatter(yteP, predP, "Prezzo MB – Pred vs Real (test)")

# Coefficienti (interpretazione grezza)
coefV = pd.Series(linV_log.coef_, index=base_cols).sort_values(key=np.abs, ascending=False)
coefP = pd.Series(ridgeP.coef_,   index=base_cols).sort_values(key=np.abs, ascending=False)
print("Top driver Volume:\n", coefV.head(8))
print("Top driver Prezzo:\n", coefP.head(8))
