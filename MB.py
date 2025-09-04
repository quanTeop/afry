# 0) Setup
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.rcParams["figure.figsize"] = (9,4)

# 1) definisco funzione che legga file

# 1.1) Funzione per leggere i file XML dei prezzi MGP e filtrare una zona
def read_gme_mgp_xml(files_pattern, zone="NORD"):
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
mgp_nord = read_gme_mgp_xml("AFRY_MB/mgp_train/*MGPPrezzi.xml", zone="NORD")

# controllo 

# print(mgp_nord.shape)
# print(mgp_nord.head())

# il data set mostra un'ora mancante in corrispondenza del cambio d'ora (4343 ore invece che 4344)

# 1.2) funzione che legga file CSV di forecast + actual demand

import pandas as pd

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

    # iv) costruzione DataFrame ordinato
    
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
