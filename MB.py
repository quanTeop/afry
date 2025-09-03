# 0) Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.rcParams["figure.figsize"] = (9,4)

# 1) Caricamento dati 
# Richiesti: MGP price, Load forecast/actual, RES forecast/actual, MB volumes & prices (Zona Nord)
mgp = pd.read_csv("MGP_prezzi.csv", parse_dates=["datetime"])
load = pd.read_csv("domanda_it.csv", parse_dates=["datetime"])          # forecast & actual
res  = pd.read_csv("res_it.csv", parse_dates=["datetime"])              # forecast & actual
mb   = pd.read_csv("mb_volumi_prezzi.csv", parse_dates=["datetime"])    # volumi & prezzi MB

# Se i file hanno colonne inutili, tieni solo ciò che serve e rinomina in modo chiaro:
# Esempio di naming atteso:
# load:  'domanda_prevista','domanda_reale'
# res:   'res_prevista','res_reale'
# mgp:   'prezzo_mgp'
# mb:    'volume_mb','prezzo_mb'

# 2) Merge sul timestamp
dfs = (mgp.set_index("datetime")
       .join(load.set_index("datetime"))
       .join(res.set_index("datetime"))
       .join(mb.set_index("datetime")))
df = dfs.reset_index().dropna().copy()

# 3) Filtra Zona Nord se i CSV sono zonali (altrimenti lascia nazionale)
# df = df[df["zona"]=="NORD"]

# 4) Feature engineering richiesto dal brief
df["delta_domanda"] = df["domanda_reale"] - df["domanda_prevista"]
df["delta_res"]     = df["res_reale"]     - df["res_prevista"]
df["hour"]          = df["datetime"].dt.hour
df["month"]         = df["datetime"].dt.month

# 5) Analisi esplorativa minima
fig, ax = plt.subplots()
ax.hist(df["delta_domanda"], bins=40); ax.set_title("Distribuzione ΔDomanda"); plt.show()

fig, ax = plt.subplots()
ax.scatter(df["delta_domanda"], df["volume_mb"], s=5, alpha=0.4)
ax.set_xlabel("ΔDomanda [MW]"); ax.set_ylabel("Volume MB [MW]")
ax.set_title("ΔDomanda vs Volume MB"); plt.show()

fig, ax = plt.subplots()
ax.scatter(df["delta_res"], df["volume_mb"], s=5, alpha=0.4, color="tab:orange")
ax.set_xlabel("ΔRES [MW]"); ax.set_ylabel("Volume MB [MW]")
ax.set_title("ΔRES vs Volume MB"); plt.show()

# 6) Split train/test (gen-giu vs lug-ago 2025)
train = df[(df["datetime"] >= "2025-01-01") & (df["datetime"] < "2025-07-01")]
test  = df[(df["datetime"] >= "2025-07-01") & (df["datetime"] < "2025-09-01")]

features = ["delta_domanda","delta_res","prezzo_mgp"]  # modello minimo richiesto
X_train, y_train = train[features], train["volume_mb"]
X_test,  y_test  = test[features],  test["volume_mb"]

# 7) Modello semplice (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test  = model.predict(X_test)

# 8) Metriche (RMSE + MAPE robusta)
rmse_train = mean_squared_error(y_train, pred_train, squared=False)
rmse_test  = mean_squared_error(y_test,  pred_test,  squared=False)

eps = 1e-6
mape_train = (np.abs((y_train - pred_train) / (np.maximum(np.abs(y_train), eps)))).mean()*100
mape_test  = (np.abs((y_test  - pred_test)  / (np.maximum(np.abs(y_test),  eps)))).mean()*100

print(f"RMSE Train: {rmse_train:.1f}  |  RMSE Test: {rmse_test:.1f}")
print(f"MAPE Train: {mape_train:.1f}% |  MAPE Test: {mape_test:.1f}%")

# 9) Plot confronto temporale sul periodo di test
fig, ax = plt.subplots()
ax.plot(test["datetime"], y_test, label="Reale", linewidth=1)
ax.plot(test["datetime"], pred_test, label="Predetto", linewidth=1, linestyle="--")
ax.set_title("Volume MB - Reale vs Predetto (Test: Lug-Ago 2025)")
ax.legend(); plt.show()

# 10) (Opzionale) Ripeti per 'prezzo_mb' come target
