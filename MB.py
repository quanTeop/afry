# 0) Setup
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
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

# 1.3) funzione che legga csv prezzi e volumi MB

import pandas as pd
import glob
import csv

def _read_mb_csv(path):
    """
    Lettore robusto per i CSV Terna (sep=',', campi quotati con ").
    """
    try:
        return pd.read_csv(path, sep=",", quotechar='"', encoding="utf-8", engine="python")
    except Exception:
        pass
    try:
        return pd.read_csv(path, sep=",", quotechar='"', encoding="latin1", engine="python")
    except Exception:
        pass
    try:
        return pd.read_csv(
            path, sep=",", encoding="utf-8", engine="python",
            quoting=csv.QUOTE_NONE, escapechar="\\", on_bad_lines="skip"
        )
    except Exception:
        pass
    return pd.read_excel(path)

def _to_num_series(s):
    s = s.astype(str)
    s = s.str.replace('n/e', '0', regex=False)
    s = s.str.replace(' ',   '', regex=False)
    s = s.str.replace('.',   '', regex=False)   # separatore migliaia
    s = s.str.replace(',',   '.', regex=False)  # separatore decimale
    return pd.to_numeric(s, errors='coerce').fillna(0)

def _parse_datetime_flex(series):
    txt = (series.astype(str)
                 .str.strip()
                 .str.replace('\u00A0', ' ', regex=False))
    if (txt.str.contains(" - ").any()):
        txt = txt.str.split(" - ", n=1).str[0]

    fmts = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]
    for f in fmts:
        dt = pd.to_datetime(txt, format=f, errors="coerce")
        if dt.notna().any():
            return dt
    return pd.to_datetime(txt, dayfirst=True, errors="coerce")

def read_mb_monthly_quarter(pattern, zone="NORD"):
    """
    Legge uno o più CSV 'Riepilogo_Mensile_Quarto_Orario_YYYYMM.csv',
    filtra Macrozona = zone, e restituisce serie ORARIE:
      datetime, sbil_netto, volume_up, volume_down, volume_mb, prezzo_mb
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Nessun file trovato per pattern: {pattern}")

    pieces = []
    for p in paths:
        df = _read_mb_csv(p)
        df.columns = [c.strip() for c in df.columns]

        cand_time  = [c for c in df.columns if "DATA" in c.upper()]
        cand_zone  = [c for c in df.columns if "MACROZONA" in c.upper()]
        cand_sbil  = [c for c in df.columns if ("SBIL" in c.upper() and "MWH" in c.upper())]
        cand_price = [c for c in df.columns if ("PREZZO" in c.upper() and "SBIL" in c.upper())]

        if not (cand_time and cand_zone and cand_sbil and cand_price):
            raise ValueError(f"Header inatteso in {p}. Colonne viste: {list(df.columns)}")

        time_col, zona_col, sbil_col, prezzo_col = cand_time[0], cand_zone[0], cand_sbil[0], cand_price[0]

        zone_up = zone.upper()
        zser = df[zona_col].astype(str).str.upper().str.strip()
        dfx = df[zser.eq(zone_up)].copy()
        if dfx.empty:
            dfx = df[zser.str.contains(zone_up, na=False)].copy()

        dt_q = _parse_datetime_flex(dfx[time_col])
        if dt_q.isna().all():
            examples = dfx[time_col].astype(str).head(3).tolist()
            raise ValueError(f"Parse datetime fallito su {time_col}. Esempi: {examples}")

        sbil_q = _to_num_series(dfx[sbil_col])     # MWh su 15'
        p_q    = _to_num_series(dfx[prezzo_col])   # €/MWh su 15'
        w      = sbil_q.abs()

        pieces.append(pd.DataFrame({
            "datetime_qh": dt_q,
            "sbil_qMWh": sbil_q,
            "price_q": p_q,
            "w": w
        }))

    qh = (pd.concat(pieces, ignore_index=True)
            .dropna(subset=["datetime_qh"])
            .sort_values("datetime_qh"))

    qh["datetime"] = qh["datetime_qh"].dt.floor("h")

    tmp = qh.copy()
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


# TRAIN = Gen–Giu
mb_gen = read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202501.csv")
mb_feb = read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202502.csv")
mb_mar = read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202503.csv")
mb_apr = read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202504.csv")
mb_mag = read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202505.csv")
mb_giu = read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202506.csv")
mb_train = pd.concat([mb_gen, mb_feb, mb_mar, mb_apr, mb_mag, mb_giu], ignore_index=True)

# TEST = Lug–Ago
mb_lug = read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202507.csv")
mb_ago = read_mb_monthly_quarter("AFRY_MB/mb/Riepilogo_Mensile_Quarto_Orario_202508.csv")
mb_test = pd.concat([mb_lug, mb_ago], ignore_index=True)

mb_all = pd.concat([mb_train, mb_test], ignore_index=True).sort_values("datetime")
print(mb_all.head(), len(mb_all))
