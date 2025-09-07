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
