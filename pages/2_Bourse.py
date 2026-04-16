# pages/2_Bourse.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

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

def make_chart(df, predictions_dict, company, months):
    fig = go.Figure()
    hist = df.tail(90)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].values.flatten(), name='📈 Historique', line=dict(color='#00ff88', width=2.5)))
    colors = {'LSTM': '#00b4d8', 'Prophet': '#ff6b35', 'NeuralProphet': '#9b59b6'}
    for name, data in predictions_dict.items():
        fig.add_trace(go.Scatter(x=data['dates'], y=data['preds'], name=f'🔮 {name}', line=dict(color=colors.get(name, '#fff'), width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[float(df['Close'].iloc[-1])], mode='markers', name="📍 Aujourd'hui", marker=dict(size=14, color='#ff4444', symbol='star')))
    fig.update_layout(title=f"📊 {company} — {months} mois", xaxis_title="Date", yaxis_title="Prix ($)", template="plotly_dark", height=500)
    return fig

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

@st.cache_data
def get_data_cached(name):
    return get_stock_data(name)

@st.cache_resource
def get_lstm_cached(name):
    return load_lstm(name)

if st.button("🚀 Lancer les prédictions", type="primary"):
    # Chargement des données avec vérification
    df = get_data_cached(company)
    
    if df is None or df.empty:
        st.error(f"❌ Impossible de charger les données pour {company}. Vérifiez votre connexion.")
        st.stop()
    
    if len(df) < 30:
        st.warning(f"⚠️ Données insuffisantes ({len(df)} jours). Minimum 30 jours requis.")
        st.stop()
    
    current_price = float(df['Close'].iloc[-1])
    st.success(f"✅ {len(df)} jours | Prix: ${current_price:.2f}")
    
    predictions_dict = {}
    
    if use_lstm:
        with st.spinner("LSTM..."):
            try:
                model = get_lstm_cached(company)
                if model:
                    dates, preds = predict_lstm(model, df, months)
                    if dates is not None:
                        predictions_dict['LSTM'] = {'dates': dates, 'preds': preds}
                        st.success("✅ LSTM")
                else:
                    st.warning("⚠️ Modèle LSTM non trouvé")
            except Exception as e:
                st.warning(f"LSTM: {e}")
    
    if use_prophet:
        with st.spinner("Prophet..."):
            try:
                model = load_prophet_from_json(company)
                if model is None:
                    model, _ = train_prophet(df)
                dates, preds = predict_prophet(model, df, months)
                predictions_dict['Prophet'] = {'dates': dates, 'preds': preds}
                st.success("✅ Prophet")
            except Exception as e:
                st.warning(f"Prophet: {e}")
    
    if use_neural and check_neuralprophet():
        with st.spinner("NeuralProphet..."):
            try:
                np_model, np_df = train_neuralprophet(df)
                dates, preds = predict_neuralprophet(np_model, np_df, df, months)
                predictions_dict['NeuralProphet'] = {'dates': dates, 'preds': preds}
                st.success("✅ NeuralProphet")
            except Exception as e:
                st.warning(f"NeuralProphet: {e}")
    
    if predictions_dict:
        st.session_state['stock_preds'] = predictions_dict
        st.session_state['stock_current_price'] = current_price
        st.session_state['stock_company'] = company
        st.session_state['stock_months'] = months

if 'stock_preds' in st.session_state:
    predictions_dict = st.session_state['stock_preds']
    current_price = st.session_state['stock_current_price']
    company = st.session_state['stock_company']
    months = st.session_state['stock_months']
    
    df = get_data_cached(company)
    if df is not None:
        fig = make_chart(df, predictions_dict, company, months)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau
        rows = []
        for name, data in predictions_dict.items():
            final = float(data['preds'][-1])
            var = (final - current_price) / current_price * 100
            rows.append({"Modèle": name, "Prix final": f"${final:.2f}", "Variation": f"{var:+.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        
        # Chatbot financier
        st.markdown("---")
        st.subheader("💬 Chatbot Financier")
        
        if 'stock_history' not in st.session_state:
            st.session_state['stock_history'] = []
        
        for msg in st.session_state['stock_history']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])
        
        user_input = st.chat_input("Posez une question...")
        if user_input:
            st.session_state['stock_history'].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            with st.spinner("Réflexion..."):
                response = chat_stock(
                    company, months,
                    float(predictions_dict.get('LSTM', {}).get('preds', [0])[-1]),
                    float(predictions_dict.get('Prophet', {}).get('preds', [0])[-1]),
                    float(predictions_dict.get('NeuralProphet', {}).get('preds', [0])[-1]),
                    st.session_state['stock_history']
                )
            st.session_state['stock_history'].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)

st.markdown("---")
st.caption("⚠️ Prédictions générées par IA - Pas un conseil financier")
