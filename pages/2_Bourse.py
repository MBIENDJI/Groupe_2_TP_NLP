# pages/2_Bourse.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import time
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMPANIES
from utils_stocks import (
    get_stock_data,
    load_lstm,
    predict_lstm,
    load_prophet_from_json,
    train_prophet,
    predict_prophet,
    check_neuralprophet,
    train_neuralprophet,
    predict_neuralprophet
)
from chatbot.llm_groq import chat_stock

st.set_page_config(page_title="Prédiction Boursière", page_icon="📈", layout="wide")

st.title("📈 Prédiction Boursière")
st.markdown("**LSTM · Prophet · NeuralProphet**")
st.markdown("---")

# ============================================================
# FALLBACK
# ============================================================

def get_fallback_data(company):
    np.random.seed(42)
    end = datetime.now()
    start = end - timedelta(days=730)
    dates = pd.bdate_range(start=start, end=end)
    base = {'NVIDIA': 178, 'ORACLE': 141, 'IBM': 240, 'CISCO': 83}.get(company, 100)
    trend = np.linspace(0, base * 0.1, len(dates))
    noise = np.random.randn(len(dates)) * base * 0.02
    prices = (base + trend + noise).astype(float)
    df = pd.DataFrame({'Close': prices}, index=dates)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

# ============================================================
# CACHE
# ============================================================

@st.cache_data(ttl=3600)
def get_data_safe(company):
    try:
        df = get_stock_data(company)
        if df is not None and not df.empty and len(df) >= 30 and 'Close' in df.columns:
            _ = float(df['Close'].values.flatten().astype(float)[-1])
            return df
    except Exception as e:
        st.warning(f"⚠️ Yahoo: {str(e)[:60]}")
    st.info(f"📊 Données simulées pour {company}")
    return get_fallback_data(company)

@st.cache_resource
def get_lstm_cached(company):
    try:
        return load_lstm(company)
    except:
        return None

# ============================================================
# GRAPHIQUE
# ============================================================

def make_chart(df, predictions_dict, company, months):
    fig = go.Figure()
    hist = df.tail(90)
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'].values.flatten().astype(float),
        name='📈 Historique',
        line=dict(color='#00ff88', width=2.5)
    ))
    colors = {'LSTM': '#00b4d8', 'Prophet': '#ff6b35', 'NeuralProphet': '#9b59b6'}
    for name, data in predictions_dict.items():
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['preds'].astype(float),
            name=f'🔮 {name}',
            line=dict(color=colors.get(name, '#fff'), width=2, dash='dash')
        ))
    last_close = float(df['Close'].values.flatten().astype(float)[-1])
    fig.add_trace(go.Scatter(
        x=[df.index[-1]],
        y=[last_close],
        mode='markers',
        name="📍 Aujourd'hui",
        marker=dict(size=14, color='#ff4444', symbol='star')
    ))
    fig.update_layout(
        title=f"📊 {company} — {months} mois",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        template="plotly_dark",
        height=500,
        hovermode='x unified'
    )
    return fig

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
# BOUTON
# ============================================================

if st.button("🚀 Lancer les prédictions", type="primary", use_container_width=True):
    with st.spinner("📡 Chargement des données..."):
        df = get_data_safe(company)

    close_arr = df['Close'].values.flatten().astype(float)
    current_price = float(close_arr[-1])
    st.success(f"✅ {len(df)} jours | Prix: **${current_price:.2f}**")

    predictions_dict = {}

    if use_lstm:
        with st.spinner("🔮 LSTM..."):
            try:
                model = get_lstm_cached(company)
                if model:
                    dates, preds = predict_lstm(model, df, months)
                    predictions_dict['LSTM'] = {'dates': dates, 'preds': np.array(preds).astype(float)}
                    st.success("✅ LSTM")
                else:
                    st.warning(f"⚠️ Modèle LSTM non trouvé")
            except Exception as e:
                st.warning(f"⚠️ LSTM: {str(e)[:80]}")

    if use_prophet:
        with st.spinner("📊 Prophet..."):
            try:
                model = load_prophet_from_json(company)
                if model is None:
                    model, _ = train_prophet(df)
                dates, preds = predict_prophet(model, df, months)
                predictions_dict['Prophet'] = {'dates': dates, 'preds': np.array(preds).astype(float)}
                st.success("✅ Prophet")
            except Exception as e:
                st.warning(f"⚠️ Prophet: {str(e)[:80]}")

    if use_neural and check_neuralprophet():
        with st.spinner("🧠 NeuralProphet (2-5 min)..."):
            try:
                np_model, np_df = train_neuralprophet(df)
                dates, preds = predict_neuralprophet(np_model, np_df, df, months)
                predictions_dict['NeuralProphet'] = {'dates': dates, 'preds': np.array(preds).astype(float)}
                st.success("✅ NeuralProphet")
            except Exception as e:
                st.warning(f"⚠️ NeuralProphet: {str(e)[:80]}")

    if predictions_dict:
        st.session_state['stock_predictions'] = predictions_dict
        st.session_state['stock_current_price'] = current_price
        st.session_state['stock_company'] = company
        st.session_state['stock_months'] = months
        st.session_state['stock_df'] = df
        st.session_state['stock_chat_history'] = []
        st.rerun()
    else:
        st.error("❌ Aucune prédiction générée")

# ============================================================
# AFFICHAGE
# ============================================================

if 'stock_predictions' in st.session_state:
    predictions_dict = st.session_state['stock_predictions']
    current_price = st.session_state['stock_current_price']
    company = st.session_state['stock_company']
    months = st.session_state['stock_months']
    df = st.session_state.get('stock_df')

    if df is not None:
        st.markdown("---")
        st.subheader("📈 Courbes des prédictions")
        fig = make_chart(df, predictions_dict, company, months)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Résultats")
        rows = []
        for name, data in predictions_dict.items():
            final = float(data['preds'].astype(float)[-1])
            variation = (final - current_price) / current_price * 100
            rows.append({"Modèle": name, "Prix prédit": f"${final:.2f}", "Variation": f"{variation:+.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Chatbot
        st.markdown("---")
        st.subheader("💬 Chatbot Financier")
        st.info(f"📊 {company} | {months} mois | Prix: ${current_price:.2f}")

        if 'stock_chat_history' not in st.session_state:
            st.session_state['stock_chat_history'] = []

        for msg in st.session_state['stock_chat_history']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

        user_input = st.chat_input("Posez une question...")
        if user_input:
            st.session_state['stock_chat_history'].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            lstm_pred = float(predictions_dict.get('LSTM', {}).get('preds', [0])[-1]) if 'LSTM' in predictions_dict else 0
            prophet_pred = float(predictions_dict.get('Prophet', {}).get('preds', [0])[-1]) if 'Prophet' in predictions_dict else 0
            neural_pred = float(predictions_dict.get('NeuralProphet', {}).get('preds', [0])[-1]) if 'NeuralProphet' in predictions_dict else 0

            with st.spinner("Réflexion..."):
                response = chat_stock(company, months, lstm_pred, prophet_pred, neural_pred, st.session_state['stock_chat_history'])
            st.session_state['stock_chat_history'].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)

st.markdown("---")
st.caption("⚠️ Prédictions IA - Pas un conseil financier")
