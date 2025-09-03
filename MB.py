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

# Funzione per leggere i file XML dei prezzi MGP e filtrare una zona
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

# Legge tutti i file XML della cartella mgp/ che terminano con MGPPrezzi.xml filtrando nodo NORD
mgp_nord = read_gme_mgp_xml("AFRY_MB/mgp_train/*MGPPrezzi.xml", zone="NORD")

print(mgp_nord.shape)
print(mgp_nord.head())

# il data set mostra un'ora mancante in corrispondenza del cambio d'ora (4343 ore invece che 4344)
