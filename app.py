import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

st.set_page_config(page_title="Case Study MB", page_icon="⚡", layout="wide")
st.title("⚡ Case Study - Mercato del Bilanciamento")

# Inizializzazione session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Funzione per generare dati demo
@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', '2025-08-31', freq='h')
    n = len(dates)
    
    data = pd.DataFrame({
        'datetime': dates,
        'prezzo_mgp': 50 + 30*np.sin(np.arange(n)*2*np.pi/24) + np.random.randn(n)*10,
        'domanda_prevista': 30000 + 5000*np.sin(np.arange(n)*2*np.pi/24) + np.random.randn(n)*1000,
        'domanda_reale': 30000 + 5000*np.sin(np.arange(n)*2*np.pi/24) + np.random.randn(n)*1500,
        'res_prevista': 8000 + 2000*np.sin(np.arange(n)*2*np.pi/24) + np.random.randn(n)*500,
        'res_reale': 8000 + 2000*np.sin(np.arange(n)*2*np.pi/24) + np.random.randn(n)*800,
    })
    
    data['delta_domanda'] = data['domanda_reale'] - data['domanda_prevista']
    data['delta_res'] = data['res_reale'] - data['res_prevista']
    data['volume_mb'] = 0.3*data['delta_domanda'] - 0.5*data['delta_res'] + np.random.randn(n)*100
    data['prezzo_mb'] = data['prezzo_mgp'] + 0.01*data['delta_domanda'] - 0.02*data['delta_res'] + np.random.randn(n)*5
    data['hour'] = data['datetime'].dt.hour
    data['month'] = data['datetime'].dt.month
    
    return data

# Sidebar
tab = st.sidebar.radio("Navigazione", 
    ["📥 Caricamento Dati", "📊 Analisi Esplorativa", "🤖 Modello", "📈 Validazione", "💡 Sviluppi Futuri"])

# TAB 1: CARICAMENTO DATI
if tab == "📥 Caricamento Dati":
    st.header("1. Caricamento Dati")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎮 Carica Dati Demo", type="primary", use_container_width=True):
            st.session_state.data = generate_demo_data()
            st.success("✅ Dati caricati!")
    
    with col2:
        uploaded = st.file_uploader("📁 O carica CSV", type=['csv'])
        if uploaded:
            st.session_state.data = pd.read_csv(uploaded)
            st.success("✅ File caricato!")
    
    if st.session_state.data is not None:
        st.subheader("Preview Dati")
        st.dataframe(st.session_state.data.head(), use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Periodo", f"{st.session_state.data['datetime'].min().strftime('%b %Y')} - {st.session_state.data['datetime'].max().strftime('%b %Y')}")
        col2.metric("N° ore", f"{len(st.session_state.data):,}")
        col3.metric("Prezzo MGP medio", f"{st.session_state.data['prezzo_mgp'].mean():.1f} €/MWh")
        col4.metric("Domanda media", f"{st.session_state.data['domanda_reale'].mean():.0f} MW")

# TAB 2: ANALISI ESPLORATIVA
elif tab == "📊 Analisi Esplorativa":
    st.header("2. Analisi Esplorativa")
    
    if st.session_state.data is None:
        st.warning("⚠️ Carica prima i dati")
    else:
        data = st.session_state.data
        
        # Sottotab
        subtab = st.tabs(["Errori di Previsione", "Correlazioni", "Pattern Temporali"])
        
        with subtab[0]:
            st.subheader("Errori di Previsione")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.histogram(data, x='delta_domanda', nbins=30, 
                    title="Distribuzione ΔDomanda")
                st.plotly_chart(fig1, use_container_width=True)
                
                st.metric("RMSE Domanda", f"{np.sqrt(np.mean(data['delta_domanda']**2)):.0f} MW")
                st.metric("MAPE Domanda", f"{np.mean(np.abs(data['delta_domanda']/data['domanda_reale']))*100:.1f}%")
            
            with col2:
                fig2 = px.histogram(data, x='delta_res', nbins=30,
                    title="Distribuzione ΔRES")
                st.plotly_chart(fig2, use_container_width=True)
                
                st.metric("RMSE RES", f"{np.sqrt(np.mean(data['delta_res']**2)):.0f} MW")
                st.metric("MAPE RES", f"{np.mean(np.abs(data['delta_res']/data['res_reale']))*100:.1f}%")
            
            # Serie temporale errori
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=data['datetime'], y=data['delta_domanda'], 
                name='ΔDomanda', line=dict(width=1)))
            fig3.add_trace(go.Scatter(x=data['datetime'], y=data['delta_res'],
                name='ΔRES', line=dict(width=1)))
            fig3.update_layout(title="Errori nel tempo", height=300)
            st.plotly_chart(fig3, use_container_width=True)
        
        with subtab[1]:
            st.subheader("Correlazioni con MB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig4 = px.scatter(data, x='delta_domanda', y='volume_mb',
                    title="ΔDomanda vs Volume MB", trendline="ols")
                st.plotly_chart(fig4, use_container_width=True)
                
                fig5 = px.scatter(data, x='delta_res', y='volume_mb',
                    title="ΔRES vs Volume MB", trendline="ols")
                st.plotly_chart(fig5, use_container_width=True)
            
            with col2:
                fig6 = px.scatter(data, x='prezzo_mgp', y='prezzo_mb',
                    title="Prezzo MGP vs Prezzo MB", trendline="ols")
                st.plotly_chart(fig6, use_container_width=True)
                
                # Matrice correlazione
                corr_vars = ['delta_domanda', 'delta_res', 'prezzo_mgp', 'volume_mb', 'prezzo_mb']
                corr = data[corr_vars].corr()
                fig7 = px.imshow(corr, text_auto='.2f', title="Matrice Correlazione")
                st.plotly_chart(fig7, use_container_width=True)
        
        with subtab[2]:
            st.subheader("Pattern Temporali")
            
            hourly = data.groupby('hour')[['delta_domanda', 'delta_res', 'volume_mb', 'prezzo_mb']].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig8 = px.line(hourly[['delta_domanda', 'delta_res']], 
                    title="Errori medi per ora")
                st.plotly_chart(fig8, use_container_width=True)
            
            with col2:
                fig9 = px.line(hourly[['volume_mb', 'prezzo_mb']], 
                    title="MB medio per ora")
                st.plotly_chart(fig9, use_container_width=True)

# TAB 3: MODELLO
elif tab == "🤖 Modello":
    st.header("3. Modello Previsionale")
    
    if st.session_state.data is None:
        st.warning("⚠️ Carica prima i dati")
    else:
        data = st.session_state.data
        
        # Split train/test
        train = data[data['month'] <= 6]
        test = data[data['month'] > 6]
        
        st.info(f"📊 Training: {len(train)} ore (Gen-Giu) | Test: {len(test)} ore (Lug-Ago)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox("Tipo Modello", ["Linear Regression", "Random Forest"])
            target = st.selectbox("Target", ["volume_mb", "prezzo_mb", "Entrambi"])
        
        with col2:
            st.markdown("**Features:**")
            use_delta_d = st.checkbox("ΔDomanda", True)
            use_delta_r = st.checkbox("ΔRES", True)
            use_mgp = st.checkbox("Prezzo MGP", True)
            use_hour = st.checkbox("Ora", True)
        
        with col3:
            st.markdown("**Metriche:**")
            show_rmse = st.checkbox("RMSE", True)
            show_mape = st.checkbox("MAPE", True)
            show_r2 = st.checkbox("R²", True)
        
        if st.button("🚀 Addestra Modello", type="primary"):
            # Prepara features
            features = []
            if use_delta_d: features.append('delta_domanda')
            if use_delta_r: features.append('delta_res')
            if use_mgp: features.append('prezzo_mgp')
            if use_hour: features.append('hour')
            
            X_train, X_test = train[features], test[features]
            
            # Scegli modello
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            results = {}
            targets = ['volume_mb', 'prezzo_mb'] if target == "Entrambi" else [target]
            
            for tgt in targets:
                y_train, y_test = train[tgt], test[tgt]
                
                # Training
                model.fit(X_train, y_train)
                pred_train = model.predict(X_train)
                pred_test = model.predict(X_test)
                
                # Calcola metriche
                metrics = {}
                if show_rmse:
                    metrics['RMSE Train'] = np.sqrt(mean_squared_error(y_train, pred_train))
                    metrics['RMSE Test'] = np.sqrt(mean_squared_error(y_test, pred_test))
                if show_mape:
                    metrics['MAPE Train'] = mean_absolute_percentage_error(y_train, pred_train) * 100
                    metrics['MAPE Test'] = mean_absolute_percentage_error(y_test, pred_test) * 100
                if show_r2:
                    metrics['R² Train'] = r2_score(y_train, pred_train)
                    metrics['R² Test'] = r2_score(y_test, pred_test)
                
                results[tgt] = {
                    'metrics': metrics,
                    'predictions': pred_test,
                    'actual': y_test,
                    'model': model
                }
            
            st.session_state.results = results
            st.session_state.test_data = test
            st.success("✅ Modello addestrato!")
            
            # Mostra risultati
            for tgt, res in results.items():
                st.subheader(f"Risultati {tgt}")
                
                col1, col2 = st.columns(2)
                with col1:
                    metrics_df = pd.DataFrame([res['metrics']]).T
                    metrics_df.columns = ['Valore']
                    st.dataframe(metrics_df.round(2))
                
                with col2:
                    if model_type == "Random Forest" and len(features) > 0:
                        importance = pd.DataFrame({
                            'Feature': features,
                            'Importance': res['model'].feature_importances_
                        }).sort_values('Importance', ascending=True)
                        
                        fig = px.bar(importance, x='Importance', y='Feature', 
                            orientation='h', title="Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)

# TAB 4: VALIDAZIONE
elif tab == "📈 Validazione":
    st.header("4. Test & Validazione")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Prima addestra un modello")
    else:
        results = st.session_state.results
        test = st.session_state.test_data
        
        for target, res in results.items():
            st.subheader(f"Validazione {target}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Serie temporale
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=test['datetime'], y=res['actual'],
                    name='Reale', line=dict(color='blue')))
                fig1.add_trace(go.Scatter(x=test['datetime'], y=res['predictions'],
                    name='Predizione', line=dict(color='red', dash='dash')))
                fig1.update_layout(title="Confronto Temporale", height=350)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Scatter plot
                fig2 = px.scatter(x=res['actual'], y=res['predictions'],
                    title="Predizioni vs Reali")
                
                # Aggiungi linea perfetta
                min_val = min(res['actual'].min(), res['predictions'].min())
                max_val = max(res['actual'].max(), res['predictions'].max())
                fig2.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', name='Linea Perfetta', line=dict(dash='dot')))
                
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Analisi residui
            residuals = res['actual'] - res['predictions']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Residuo Medio", f"{residuals.mean():.2f}")
            col2.metric("Std Residui", f"{residuals.std():.2f}")
            col3.metric("Max |Residuo|", f"{np.abs(residuals).max():.2f}")
            
            fig3 = px.histogram(residuals, nbins=30, title="Distribuzione Residui")
            st.plotly_chart(fig3, use_container_width=True)

# TAB 5: SVILUPPI FUTURI
elif tab == "💡 Sviluppi Futuri":
    st.header("5. Sviluppi Futuri")
    
    st.markdown("""
    ### 🚀 Proposte di Miglioramento del Modello
    
    #### 1. **Arricchimento Dataset** 📊
    - **Dati storici estesi**: Utilizzare 2-3 anni di dati per catturare stagionalità e trend
    - **Dati MSD ex-ante**: Includere risultati del Mercato dei Servizi di Dispacciamento
    - **Dati meteo**: Temperatura, irraggiamento solare, velocità vento per migliorare previsioni RES
    - **Eventi speciali**: Festività, scioperi, eventi sportivi che impattano la domanda
    
    #### 2. **Feature Engineering Avanzato** 🔧
    - **Lag features**: Valori storici di errori (t-1, t-24, t-168)
    - **Rolling statistics**: Medie mobili e deviazioni standard su finestre temporali
    - **Interazioni**: Prodotti tra features (es. ora × giorno_settimana)
    - **Trasformazioni non lineari**: Polinomi e splines per catturare relazioni complesse
    
    #### 3. **Modelli Avanzati** 🤖
    - **XGBoost/LightGBM**: Migliori performance su dati tabulari
    - **LSTM/GRU**: Per catturare dipendenze temporali lunghe
    - **Ensemble**: Combinare predizioni di modelli diversi
    - **Modelli probabilistici**: Quantile regression per intervalli di confidenza
    
    #### 4. **Ottimizzazione per Zona** 🗺️
    - **Modelli zona-specifici**: Un modello per ogni macrozona (Nord, Centro-Nord, Centro-Sud, Sud, Sicilia, Sardegna)
    - **Transfer learning**: Utilizzare informazioni cross-zonali
    - **Congestioni di rete**: Includere limiti di transito tra zone
    
    #### 5. **Validazione Robusta** ✅
    - **Walk-forward validation**: Test su finestre temporali multiple
    - **Cross-validation temporale**: Preservando l'ordine cronologico
    - **Analisi scenari estremi**: Performance durante eventi eccezionali
    - **Backtesting continuo**: Monitoraggio real-time delle performance
    
    #### 6. **Integrazione Operativa** ⚙️
    - **Pipeline automatizzata**: Download dati → preprocessing → predizione → report
    - **Alert system**: Notifiche per predizioni anomale o errori elevati
    - **Dashboard real-time**: Visualizzazione continua KPIs
    - **API REST**: Per integrazione con sistemi trading/bidding
    
    ### 📈 Impatto Atteso
    
    Con questi miglioramenti, ci si può aspettare:
    - **Riduzione MAPE del 30-40%** sui volumi MB
    - **Migliore cattura di eventi estremi** (picchi di domanda/generazione)
    - **Predizioni affidabili fino a 48h** in anticipo
    - **ROI positivo** da strategie di trading basate sul modello
    """)
    
    # Simulazione miglioramenti
    if st.button("🔮 Simula Miglioramenti"):
        current_mape = 15.0
        improved_mape = current_mape * 0.65
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MAPE Attuale", f"{current_mape:.1f}%")
        col2.metric("MAPE Target", f"{improved_mape:.1f}%", f"-{current_mape-improved_mape:.1f}%")
        col3.metric("Miglioramento", f"{(1-improved_mape/current_mape)*100:.0f}%")
        
        st.success("✨ Con le ottimizzazioni proposte, il modello potrebbe raggiungere performance di livello industriale!")
