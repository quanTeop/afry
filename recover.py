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
from sklearn.metrics import root_mean_squared_error

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

            if zone.upper() not in rec:   # se manca la zona, salto: cuore del filtro
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

# provo 
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

    # v) passo da quartoraio a orario facendo la media (sto lavorando in MW)
    
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

train = load_2025[load_2025["month"] <= 6].copy()                                # Gen–Giu
test  = load_2025[(load_2025["month"] >= 7) & (load_2025["month"] <= 8)].copy()  # Lug–Ago

# provo 
# print(load_2025.head())
# print("TRAIN ore:", len(train), " | TEST ore:", len(test))
# ok

# -----------------------------------------------------
# 1.2.b) funzione che legga file CSV di forecast + actual generation ("gemella" di 1.2.a)
# -----------------------------------------------------

def read_res_total(path):
    
    # i) leggo il CSV con separatore ';'
    
    df = pd.read_csv(path, sep=';')

    # ii) prendo solo l'inizio dell'intervallo "start - end"
    
    start = df["MTU (CET/CEST)"].astype(str).str.split(" - ", n=1).str[0]
    dt = pd.to_datetime(start, dayfirst=True, errors="coerce")

    # iii) "n/e" nel csv continuava a dare errore, lo trasformo direttamente in zero
    
    def to_num(col):
        return pd.to_numeric(df[col].replace("n/e", 0), errors="coerce").fillna(0)

    # iv) devo fare una selezione delle colonne (alcune non mi interessavano). Poi creo dataframe
    
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

    # v) passo da quartoraio a orario facendo la media (sto lavorando in MW)
    
    out = (out.set_index("datetime")
              .resample("h").mean()
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

# 0) converto numeri da formato italiano: i file TERNA hanno numeri scritti come "10.000,00", ENTSO-E non avevano ne "." ne ","

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

        # ii) filtro NORD robusto (mi dava problemi)
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

# unisco in mb all

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

# cicliche: la regressione dava problemi, provo a rendere il tempo "circolare": rappresento ore e mesi come angoli su una circonferenza
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
# pesi per valutare il prezzo
wP = feat.loc[test_mask & feat["prezzo_mb"].notna(), "volume_mb"].to_numpy()  # peso per rmse (vedi sotto): prendo solo ore di test con prezzo mb disponibile e ci faccio il vettore dei pesi

# print("Shapes → Volume", XtrV.shape, XteV.shape, " | Prezzo", XtrP.shape, XteP.shape)

# ============================================
# 3) calcolo RMSE
# ============================================

def eval_rmse(y_true, y_pred, label="", weights=None):
    # RMSE: per il volume, rmse semplice. per il prezzo, meglio pesato!!! se errore sul prezzo è alto ma volume associato basso, non ha senso che gonfi l'RMSE
    rmse = root_mean_squared_error(y_true, y_pred, sample_weight=weights)
    print(f"[{label}] RMSE={rmse:,.1f}")
    return rmse
    
# stima del Volume: regressione lineare 
linV = LinearRegression().fit(XtrV, ytrV)                  # alleno modello lineare sui dati train
predV = linV.predict(XteV)                                 # creo+ un vettore previsione del target
rmse_V = eval_rmse(yteV, predV, "Volume (lin)")            # calcolo RMSE fra test e previsione

# stima del Prezzo: regressione lineare (uguale ma con peso)
linP  = LinearRegression().fit(XtrP, ytrP)
predP = linP.predict(XteP)
rmse_P = eval_rmse(yteP, predP, "Prezzo (lin)", weights=wP)           # calcolo RMSE fra test e previsione

# sto comparando due rmse (uno sul volume, uno sul prezzo) con due unità di misura diverse
# non ha senso fisico la cosa. divido allora per la media del dato TEST a cui ciascun rmse fa riferimento

mu_V = float(np.nanmean(yteV))
mu_P = float(np.average(yteP, weights=wP))     # pesata, perché non tornava quella semplice
rmse_vero_V = rmse_V / mu_V
rmse_vero_P = rmse_P / mu_P
print(f"{rmse_vero_V:.2f} {rmse_vero_P:.2f}")

# rmse_vero_V = 0.77 mentre rmse_vero_P = 0.67 con mu_P semplice
# rmse_vero_V = 0.77 mentre rmse_vero_P = 0.81 con mu_P PESATO. già meglio, mi aspettavo di predire meglio i volumi rispetto ai prezzi

# ============================================
# 4) GRAFICI
# ============================================

# plot dati reali e stimati per volumi e prezzi MB

def plot_ts(dt, y_true, y_pred, title, ylabel):
    plt.figure(figsize=(10,3.5))
    plt.plot(dt, y_true, label="Real")
    plt.plot(dt, y_pred, label="Pred")
    plt.title(title); plt.ylabel(ylabel); plt.xlabel("")
    plt.legend(); plt.tight_layout(); plt.show()

plot_ts(dt_test_V, yteV, predV, "Volume MB – Test (real vs pred)", "MWh")
plot_ts(dt_test_P, yteP, predP, "Prezzo MB – Test (real vs pred)", "€/MWh")

# scatter per legare Delta domanda e Delta res a volumi e prezzi MB

def quick_scatter(feat, mask):
    pairs = [("delta_domanda","volume_mb"),("delta_res","volume_mb"),
             ("delta_domanda","prezzo_mb"),("delta_res","prezzo_mb")]
    for x,y in pairs:
        s = feat.loc[mask,[x,y]].dropna()
        if len(s)==0: continue                 # loop sulle coppie definite prima, saltando eventuali NaN
        r = np.corrcoef(s[x], s[y])[0,1]       # correlazione di pearson per capire linearità fra i dati
        plt.figure(figsize=(4,4))
        plt.scatter(s[x], s[y], s=6, alpha=0.35)
        plt.title(f"{x} vs {y}  (r={r:.2f})"); plt.xlabel(x); plt.ylabel(y)
        plt.tight_layout(); plt.show()

quick_scatter(feat, test_mask)  

# errore percetuale medio res rispetto generazione

res_tmp = res_2025.copy()
res_tmp["perc_err_RES"] = np.where(res_tmp["generation_forecast"]>0,
                                   np.abs(res_tmp["delta_res"]) / res_tmp["generation_forecast"],
                                   np.nan)             # vettore di errori percentuali. se il forecast gen è nullo, mette nan, così non ho divisioni per zero
print("Errore percentuale medio RES (MAPE):", 100*np.nanmean(res_tmp["perc_err_RES"]), "%") # faccio la media dei dati del vettore precedente, ignorando i nan, e faccio la percentuale.

