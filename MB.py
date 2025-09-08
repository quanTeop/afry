# 0) Setup
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import re
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


plt.rcParams["figure.figsize"] = (9,4)

# =====================================================
# 1) LETTURA FILE: definizione funzioni (1.1, 1.2, 1.3)
# =====================================================

# -----------------------------------------------------
# 1.1) Funzione per leggere i file XML dei prezzi MGP e filtrare una zona
# -----------------------------------------------------

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

# -----------------------------------------------------
# 1.2.a) funzione che legga file CSV di forecast + actual demand
# -----------------------------------------------------

def read_load(path):
    
    # i) leggo il CSV con separatore ';' 
    
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

# -----------------------------------------------------
# 1.2.b) funzione che legga file CSV di forecast + actual generation # chiedo scusa per i tre livelli di nomenclatura, avevo dimenticato di numerare la sezione
# -----------------------------------------------------

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

# -----------------------------------------------------
# 1.3) funzione che legga file CSV di volumi e prezzi MB
# -----------------------------------------------------

# 0) converto numeri da formato italiano
def _it_num(x):
    s = pd.Series(x, dtype="string")
    s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

def read_mb_files(paths, zone="NORD"):

    paths = [str(p) for p in paths]
    frames = []

    for p in paths:
        df = pd.read_csv(p, sep=';', engine='python', dtype=str)

        # i) nomi colonne 
        tcol  = "Data Riferimento"
        zcol  = "Macrozona"
        sbcol = "Sbil aggregato zonale [MWh]"
        sgcol = "Segno aggregato zonale"
        pcol  = "Prezzo di sbilanciamento"  # lo uso direttamente come totale

        # ii) filtro NORD
        dfx = df[df[zcol].str.upper().str.contains(zone.upper(), na=False)].copy()
        if dfx.empty:
            continue

        # iii) tempo (inizio del quarto)
        dt_q = pd.to_datetime(dfx[tcol], dayfirst=True, errors="coerce")

        # iv) sbilancio con segno (QUARTO!!!)
        mag   = _it_num(dfx[sbcol])                   # MWh per quarto!
        segno = dfx[sgcol].str.strip().map({'+': 1.0, '-': -1.0}).fillna(1.0)
        sbil_q = mag * segno

        # v) prezzo di quarto 
        price_q = _it_num(dfx[pcol])
        
        # vi) per ogni mese, creo un mini dataframe a tre colonne. un elemento di "frames" è un mese
        frames.append(pd.DataFrame({
            "datetime_qh": dt_q,
            "sbil_qMWh": sbil_q,
            "price_q": price_q
        }))

    if not frames:
        raise ValueError("Nessun dato MB trovato. Controlla percorsi/zone.")
        
    # vii) unisco i mini dataframe
    qh = (pd.concat(frames, ignore_index=True)
            .dropna(subset=["datetime_qh"])
            .sort_values("datetime_qh"))

    # viii) passo a ORARIO. somma dei 4 quarti per i volumi; prezzo medio pesato |sbil|
    qh["datetime"] = qh["datetime_qh"].dt.floor("h") # floor mi appiattisce all'ora. i 4 quarti della stessa ora hanno uguale datetime (USO GROUPBY SOTTO)
    qh["w"] = qh["sbil_qMWh"].abs()                  # prendo lo sbilanciamento come peso per cui moltiplico il prezzo del quarto associato
    qh["wprice"] = qh["price_q"] * qh["w"]
    grp = qh.groupby("datetime", sort=True)
    sbil_netto  = grp["sbil_qMWh"].sum()
    volume_mb  = grp["sbil_qMWh"].apply(lambda s: s.abs().sum())
    num         = grp["wprice"].sum()
    den         = grp["w"].sum()
    prezzo_mb   = num.div(den).where(den > 0, grp["price_q"].mean())   # se non c'è sbilanciamento, media semplice dei quarti. non so se capiti, ma avevo paura di una divisione per zero

    out = (pd.DataFrame({
        "datetime": sbil_netto.index,
        "sbil_netto": sbil_netto.values,
        "prezzo_mb": prezzo_mb.values,
        "volume_mb": volume_mb.values
    })
    .reset_index(drop=True))
    out["month"] = out["datetime"].dt.month
    return out

# UTILIZZO LA FUNZIONE

base = Path("AFRY_MB/mb")
files_train = [base/f"Riepilogo_Mensile_Quarto_Orario_2025{m:02d}.csv" for m in range(1,7)]
files_test  = [base/f"Riepilogo_Mensile_Quarto_Orario_2025{m:02d}.csv" for m in (7,8)]

mb_train = read_mb_files(files_train, zone="NORD")
mb_test  = read_mb_files(files_test,  zone="NORD")

mb_all = (pd.concat([mb_train, mb_test], ignore_index=True)
            .sort_values("datetime")
            .reset_index(drop=True))

# =====================================================
# 2) unisco i dati
# =====================================================

# unisco le tabelle 
feat = (mb_all
    .merge(mgp_nord, on="datetime", how="left")                              
    .merge(load_2025[["datetime","delta_domanda"]], on="datetime", how="left")
    .merge(res_2025[["datetime","delta_res"]],       on="datetime", how="left"))

# calendario
feat["hour"]  = feat["datetime"].dt.hour
feat["dow"]   = feat["datetime"].dt.dayofweek
feat["month"] = feat["datetime"].dt.month
feat["is_we"] = (feat["dow"]>=5).astype(int)

# cicliche: la regressione dava problemi, provo a rendere il tempo "circolare"
feat["hour_sin"] = np.sin(2*np.pi*feat["hour"]/24)
feat["hour_cos"] = np.cos(2*np.pi*feat["hour"]/24)
feat["mth_sin"]  = np.sin(2*np.pi*(feat["month"]-1)/12)
feat["mth_cos"]  = np.cos(2*np.pi*(feat["month"]-1)/12)

# definisco delta netto
feat["delta_net"] = feat["delta_domanda"] - feat["delta_res"] # domanda più alta del previsto e RES più bassa del previsto: delta_net positivo
feat["abs_net"]   = feat["delta_net"].abs()
feat["pos_net"]   = feat["delta_net"].clip(lower=0.0)
feat["neg_net"]   = (-feat["delta_net"].clip(upper=0.0))

# rimuovo righe con prezzo <=0: se il volume è zero, il peso è nullo, mentre se è diverso da zero, è strano (penso riga errata)
feat.loc[feat["prezzo_mb"]<=0, "prezzo_mb"] = np.nan

#separo train e test
train_mask = feat["month"]<=6
test_mask  = (feat["month"]>=7) & (feat["month"]<=8)

base_cols = [
    "prezzo_mgp","delta_domanda","delta_res",
    "delta_net","abs_net","pos_net","neg_net",
    "hour_sin","hour_cos","mth_sin","mth_cos","is_we"
]

# riempio righe NaN DEI FEATURE di test e train senza usare il train (se no non ha senso la regressione su quei dati): uso la mediana del train # per ogni colonna c
for c in ["prezzo_mgp","delta_domanda","delta_res","delta_net","abs_net","pos_net","neg_net"]: 
    med = feat.loc[train_mask, c].median()
    feat[c] = feat[c].fillna(med)

# ora creo matrici: X saranno le colonne base, Y prima volumi mb e poi prezzi mb

# VOLUME
colsV = base_cols + ["volume_mb"]
trV = feat.loc[train_mask, ["datetime"]+colsV].dropna(subset=["volume_mb"])
teV = feat.loc[test_mask,  ["datetime"]+colsV].dropna(subset=["volume_mb"])
XtrV, XteV = trV[base_cols].to_numpy(), teV[base_cols].to_numpy() # matrici FEATURE
ytrV, yteV = trV["volume_mb"].to_numpy(), teV["volume_mb"].to_numpy() # vettori TARGET
dt_test_V  = teV["datetime"].to_numpy()

# PREZZO
colsP = base_cols + ["prezzo_mb"]
trP = feat.loc[train_mask, ["datetime"]+colsP].dropna(subset=["prezzo_mb"])
teP = feat.loc[test_mask,  ["datetime"]+colsP].dropna(subset=["prezzo_mb"])
XtrP, XteP = trP[base_cols].to_numpy(), teP[base_cols].to_numpy() # XtrV,XteV e XtrP,XteP coincidono se i dati fossero completti, niente NaN ne in volumi ne in prezzi. (se manca qualcosa, dropna scarta la riga. RICORDA: avevi riempito i NaN dei FEATURE, mai dei TARGET)
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
