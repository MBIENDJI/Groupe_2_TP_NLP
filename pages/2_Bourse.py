# pages/2_Bourse.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMPANIES
from utils_stocks import (
    get_stock_data, load_lstm, predict_lstm,
    load_prophet_from_json, train_prophet, predict_prophet,
    check_neuralprophet, train_neuralprophet, predict_neuralprophet
)
from chatbot.llm_groq import chat_stock

st.set_page_config(page_title="Prédiction Boursière", page_icon="📈", layout="wide")

st.title("📈 Prédiction Boursière")
st.markdown("**LSTM · Prophet · NeuralProphet**")
st.markdown("---")

# ============================================================
# FONCTIONS
# ============================================================

def make_chart(df, predictions_dict, company, months):
    """Crée le graphique des prédictions"""
    fig = go.Figure()
    
    # Historique (90 derniers jours)
    hist = df.tail(90)
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'].values.flatten(),
        name='📈 Historique',
        line=dict(color='#00ff88', width=2.5)
    ))
    
    # Couleurs des modèles
    colors = {'LSTM': '#00b4d8', 'Prophet': '#ff6b35', 'NeuralProphet': '#9b59b6'}
    
    # Prédictions
    for name, data in predictions_dict.items():
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['preds'],
            name=f'🔮 {name}',
            line=dict(color=colors.get(name, '#ffffff'), width=2, dash='dash')
        ))
    
    # Point aujourd'hui
    fig.add_trace(go.Scatter(
        x=[df.index[-1]],
        y=[float(df['Close'].iloc[-1])],
        mode='markers',
        name="📍 Aujourd'hui",
        marker=dict(size=14, color='#ff4444', symbol='star')
    ))
    
    fig.update_layout(
        title=f"📊 {company} — {months} mois",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0.2)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500,
        hovermode='x unified'
    )
    
    return fig

def compute_metrics(actual, predicted):
    """Calcule MAPE et RMSE"""
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    n = min(len(actual), len(predicted))
    if n == 0:
        return 999.0, 999.0
    actual = actual[-n:]
    predicted = predicted[-n:]
    mask = actual != 0
    if np.sum(mask) == 0:
        return 999.0, 999.0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    rmse = np.sqrt(np.mean((actual[-n:] - predicted[-n:]) ** 2))
    return float(mape), float(rmse)

# ============================================================
# SÉLECTEURS
# ============================================================

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox("🏢 Entreprise", list(COMPANIES.keys()))
with col2:
    months = st.slider("📅 Période (mois)", 3, 12, 6)

c1, c2, c3 = st.columns(3)
with c1:
    use_lstm = st.checkbox("✅ LSTM", value=True)
with c2:
    use_prophet = st.checkbox("✅ Prophet", value=True)
with c3:
    use_neural = st.checkbox("🧠 NeuralProphet", value=False)

# ============================================================
# CACHE
# ============================================================

@st.cache_data(ttl=3600)
def get_data_cached(name):
    """Données avec cache et gestion d'erreur"""
    try:
        df = get_stock_data(name)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        print(f"Erreur get_data_cached: {e}")
        return None

@st.cache_resource
def get_lstm_cached(name):
    try:
        return load_lstm(name)
    except:
        return None

# ============================================================
# BOUTON PRÉDICTION
# ============================================================

if st.button("🚀 Lancer les prédictions", type="primary", use_container_width=True):
    
    # 1. CHARGEMENT DES DONNÉES AVEC VÉRIFICATION
    with st.spinner("📡 Téléchargement des données..."):
        df = get_data_cached(company)
    
    # VÉRIFICATION CRITIQUE
    if df is None:
        st.error(f"❌ Impossible de charger les données pour {company}.")
        st.info("💡 Vérifiez votre connexion internet ou réessayez dans quelques minutes.")
        st.stop()
    
    if df.empty:
        st.error(f"❌ Aucune donnée trouvée pour {company}.")
        st.stop()
    
    if len(df) < 30:
        st.warning(f"⚠️ Données insuffisantes: {len(df)} jours seulement. Minimum 30 jours requis.")
        st.stop()
    
    # Vérifier la colonne Close
    if 'Close' not in df.columns:
        st.error(f"❌ Colonne 'Close' manquante pour {company}.")
        st.stop()
    
    current_price = float(df['Close'].iloc[-1])
    st.success(f"✅ Données chargées: {len(df)} jours | Prix actuel: **${current_price:.2f}**")
    
    # 2. PRÉDICTIONS
    predictions_dict = {}
    
    # LSTM
    if use_lstm:
        with st.spinner("🔮 Prédiction LSTM..."):
            try:
                model = get_lstm_cached(company)
                if model:
                    dates, preds = predict_lstm(model, df, months)
                    if dates is not None and preds is not None:
                        predictions_dict['LSTM'] = {'dates': dates, 'preds': preds}
                        st.success("✅ LSTM terminé")
                    else:
                        st.warning("⚠️ LSTM: prédiction vide")
                else:
                    st.warning("⚠️ Modèle LSTM non trouvé")
            except Exception as e:
                st.warning(f"⚠️ LSTM erreur: {str(e)[:100]}")
    
    # Prophet
    if use_prophet:
        with st.spinner("📊 Prédiction Prophet..."):
            try:
                model = load_prophet_from_json(company)
                if model is None:
                    st.info("📊 Prophet: entraînement en cours...")
                    model, _ = train_prophet(df)
                dates, preds = predict_prophet(model, df, months)
                predictions_dict['Prophet'] = {'dates': dates, 'preds': preds}
                st.success("✅ Prophet terminé")
            except Exception as e:
                st.warning(f"⚠️ Prophet erreur: {str(e)[:100]}")
    
    # NeuralProphet
    if use_neural and check_neuralprophet():
        with st.spinner("🧠 Prédiction NeuralProphet (peut prendre 1-2 min)..."):
            try:
                np_model, np_df = train_neuralprophet(df)
                dates, preds = predict_neuralprophet(np_model, np_df, df, months)
                predictions_dict['NeuralProphet'] = {'dates': dates, 'preds': preds}
                st.success("✅ NeuralProphet terminé")
            except Exception as e:
                st.warning(f"⚠️ NeuralProphet erreur: {str(e)[:100]}")
    
    # 3. AFFICHAGE DES RÉSULTATS
    if predictions_dict:
        st.session_state['stock_predictions'] = predictions_dict
        st.session_state['stock_current_price'] = current_price
        st.session_state['stock_company'] = company
        st.session_state['stock_months'] = months
        st.session_state['stock_df'] = df
        st.rerun()
    else:
        st.error("❌ Aucune prédiction générée. Vérifiez les modèles.")

# ============================================================
# AFFICHAGE DES RÉSULTATS (après prédiction)
# ============================================================

if 'stock_predictions' in st.session_state:
    predictions_dict = st.session_state['stock_predictions']
    current_price = st.session_state['stock_current_price']
    company = st.session_state['stock_company']
    months = st.session_state['stock_months']
    df = st.session_state.get('stock_df')
    
    if df is not None:
        # GRAPHIQUE
        st.markdown("---")
        st.subheader("📈 Courbes des prédictions")
        fig = make_chart(df, predictions_dict, company, months)
        st.plotly_chart(fig, use_container_width=True)
        
        # STATISTIQUES ET MÉTRIQUES
        st.subheader("📊 Comparaison des modèles")
        
        # Calculer les métriques
        hist_prices = df['Close'].values.flatten()[-60:]
        
        rows = []
        cols = st.columns(len(predictions_dict) + 1)
        cols[0].metric("💰 Prix actuel", f"${current_price:.2f}")
        
        for i, (name, data) in enumerate(predictions_dict.items()):
            preds = data['preds']
            final_price = float(preds[-1])
            variation = (final_price - current_price) / current_price * 100
            
            # MAPE sur les 60 premiers jours de prédiction
            mape, rmse = compute_metrics(hist_prices, preds[:60])
            
            rows.append({
                "Modèle": name,
                "Prix prédit": f"${final_price:.2f}",
                "Variation": f"{variation:+.1f}%",
                "MAPE": f"{mape:.1f}%",
                "RMSE": f"${rmse:.2f}"
            })
            
            cols[i + 1].metric(
                f"🔮 {name}",
                f"${final_price:.2f}",
                f"{variation:+.1f}%"
            )
        
        # Tableau détaillé
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        
        # Graphique barres
        st.subheader("📊 Comparaison visuelle des prix finaux")
        fig_bar = go.Figure()
        for name, data in predictions_dict.items():
            final = float(data['preds'][-1])
            fig_bar.add_trace(go.Bar(
                x=[name],
                y=[final],
                text=f"${final:.2f}",
                textposition='auto',
                name=name
            ))
        fig_bar.add_trace(go.Bar(
            x=["Prix actuel"],
            y=[current_price],
            text=f"${current_price:.2f}",
            textposition='auto',
            name="Prix actuel",
            marker_color='#00ff88'
        ))
        fig_bar.update_layout(
            title="Comparaison des prix finaux",
            yaxis_title="Prix ($)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Évolution mensuelle
        st.subheader("📅 Évolution mensuelle détaillée")
        evo_rows = []
        for name, data in predictions_dict.items():
            preds = data['preds']
            row = {"Modèle": name}
            step = 21  # 1 mois ≈ 21 jours
            for m in range(0, min(len(preds), step * 4), step):
                month_num = m // step + 1
                row[f"Mois {month_num}"] = f"${preds[min(m, len(preds)-1)]:.2f}"
            row[f"Mois {months}"] = f"${preds[-1]:.2f}"
            evo_rows.append(row)
        st.dataframe(pd.DataFrame(evo_rows), use_container_width=True, hide_index=True)
        
        # CHATBOT FINANCIER
        st.markdown("---")
        st.subheader("💬 Chatbot Financier Intelligent")
        st.info(f"📊 **{company}** | {months} mois | Prix actuel: **${current_price:.2f}**")
        
        if 'stock_chat_history' not in st.session_state:
            st.session_state['stock_chat_history'] = []
        
        for msg in st.session_state['stock_chat_history']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])
        
        user_input = st.chat_input("Posez une question sur cette action...")
        if user_input:
            st.session_state['stock_chat_history'].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.spinner("🤔 Réflexion..."):
                lstm_pred = float(predictions_dict.get('LSTM', {}).get('preds', [0])[-1]) if 'LSTM' in predictions_dict else 0
                prophet_pred = float(predictions_dict.get('Prophet', {}).get('preds', [0])[-1]) if 'Prophet' in predictions_dict else 0
                neural_pred = float(predictions_dict.get('NeuralProphet', {}).get('preds', [0])[-1]) if 'NeuralProphet' in predictions_dict else 0
                
                response = chat_stock(
                    company, months,
                    lstm_pred, prophet_pred, neural_pred,
                    st.session_state['stock_chat_history']
                )
            
            st.session_state['stock_chat_history'].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)

# Footer
st.markdown("---")
st.caption("⚠️ Ces prédictions sont générées par des modèles d'intelligence artificielle. Elles ne constituent pas un conseil financier. Investissez prudemment.")
