# pages/2_Bourse.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import time

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


def compute_mape(actual, predicted):
    actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
    n = min(len(actual), len(predicted))
    actual, predicted = actual[-n:], predicted[-n:]
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if np.any(mask) else 999.0


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
    use_neural = st.checkbox("🧠 NeuralProphet (lent)", value=False)

neural_ok = check_neuralprophet()
if use_neural and not neural_ok:
    st.error("⚠️ NeuralProphet incompatible avec ce serveur. Utilisez LSTM et Prophet.")
    use_neural = False


@st.cache_resource
def get_lstm_cached(name):
    return load_lstm(name)

@st.cache_data
def get_data_cached(name):
    return get_stock_data(name)


if st.button("🚀 Lancer les prédictions", type="primary"):
    df = get_data_cached(company)
    current_price = float(df['Close'].iloc[-1])
    st.success(f"✅ {len(df)} jours | Prix: ${current_price:.2f}")

    predictions_dict = {}

    if use_lstm:
        with st.spinner("LSTM..."):
            try:
                dates, preds = predict_lstm(get_lstm_cached(company), df, months)
                predictions_dict['LSTM'] = {'dates': dates, 'preds': preds}
                st.success("✅ LSTM")
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

    if use_neural:
        st.info("⏳ NeuralProphet: entraînement en cours (2-5 min)...")
        with st.spinner("NeuralProphet..."):
            try:
                np_model, np_df = train_neuralprophet(df)
                dates, preds = predict_neuralprophet(np_model, np_df, df, months)
                predictions_dict['NeuralProphet'] = {'dates': dates, 'preds': preds}
                st.success("✅ NeuralProphet")
            except Exception as e:
                st.warning(f"NeuralProphet: {e}")

    if predictions_dict:
        st.session_state['stock_context'] = {
            'company': company, 'months': months, 'current': current_price,
            'lstm_pred': float(predictions_dict['LSTM']['preds'][-1]) if 'LSTM' in predictions_dict else 0,
            'prophet_pred': float(predictions_dict['Prophet']['preds'][-1]) if 'Prophet' in predictions_dict else 0,
            'neural_pred': float(predictions_dict['NeuralProphet']['preds'][-1]) if 'NeuralProphet' in predictions_dict else 0,
        }
        st.session_state['stock_preds'] = predictions_dict
        if 'stock_history' not in st.session_state:
            st.session_state['stock_history'] = []


if 'stock_preds' in st.session_state:
    ctx = st.session_state['stock_context']
    predictions_dict = st.session_state['stock_preds']
    current_price = ctx['current']
    df_cached = get_data_cached(ctx['company'])

    st.markdown("---")
    fig = make_chart(df_cached, predictions_dict, ctx['company'], ctx['months'])
    st.plotly_chart(fig, use_container_width=True)

    # Métriques
    st.subheader("📊 Comparaison")
    cols = st.columns(len(predictions_dict) + 1)
    cols[0].metric("💵 Prix actuel", f"${current_price:.2f}")
    for i, (name, data) in enumerate(predictions_dict.items()):
        final = float(data['preds'][-1])
        delta = ((final / current_price) - 1) * 100
        cols[i + 1].metric(f"🔮 {name}", f"${final:.2f}", f"{delta:+.1f}%")

    # Tableau
    rows = []
    for name, data in predictions_dict.items():
        final = float(data['preds'][-1])
        var = (final - current_price) / current_price * 100
        rows.append({"Modèle": name, "Prix final": f"${final:.2f}", "Variation": f"{var:+.1f}%", "Signal": "🟢 ACHAT" if var > 5 else "🟡 ATTENDRE" if var > 0 else "🔴 VENDRE"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Chatbot financier
    st.markdown("---")
    st.subheader("💬 Chatbot Financier")
    st.info(f"📊 {ctx['company']} | {ctx['months']} mois | Prix: ${ctx['current']:.2f}")

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
                ctx['company'], ctx['months'],
                ctx['lstm_pred'], ctx['prophet_pred'], ctx['neural_pred'],
                st.session_state['stock_history']
            )
        st.session_state['stock_history'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
