# pages/2_Bourse.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from config       import COMPANIES
from utils_stocks import (get_stock_data, load_lstm,
                           predict_lstm,
                           load_prophet_from_json,
                           train_prophet,
                           predict_prophet,
                           train_neuralprophet,
                           predict_neuralprophet)
from chatbot.llm  import chat_stock

if not hasattr(np, 'NaN'):
    np.NaN = np.nan

st.set_page_config(
    page_title = "Prédiction Boursière",
    page_icon  = "📈",
    layout     = "wide"
)

st.title("📈 Prédiction Boursière")
st.markdown(
    "Prédictions **LSTM** + **Prophet** + **NeuralProphet**")
st.markdown("---")


# ============================================================
# SÉLECTEURS
# ============================================================

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox(
        "🏢 Entreprise",
        list(COMPANIES.keys())
    )
with col2:
    months = st.slider(
        "📅 Période de prédiction (mois)",
        min_value = 3,
        max_value = 12,
        value     = 6
    )


# ============================================================
# CHARGEMENT LSTM (cache par entreprise)
# ============================================================

@st.cache_resource
def get_lstm(name):
    return load_lstm(name)

@st.cache_data
def get_data(name):
    return get_stock_data(name)


# ============================================================
# BOUTON PRÉDICTION
# ============================================================

if st.button("🚀 Lancer les prédictions", type="primary"):

    with st.spinner("Téléchargement données..."):
        df = get_data(company)

    current_price = float(df['Close'].iloc[-1])

    # --- LSTM ---
    with st.spinner("Prédiction LSTM..."):
        lstm_model       = get_lstm(company)
        dates_l, pred_l  = predict_lstm(
            lstm_model, df, months)

    # --- Prophet ---
    with st.spinner("Prophet..."):
        # Essayer JSON d'abord, sinon entraîner
        p_model = load_prophet_from_json(company)
        if p_model is None:
            p_model, _ = train_prophet(df)
        dates_p, pred_p = predict_prophet(
            p_model, df, months)

    # --- NeuralProphet ---
    with st.spinner("NeuralProphet (1-2 min)..."):
        np_model, np_df = train_neuralprophet(df)
        dates_np, pred_np = predict_neuralprophet(
            np_model, np_df, df, months)

    # --- Stocker en session ---
    st.session_state['stock_context'] = {
        'company'     : company,
        'months'      : months,
        'current'     : current_price,
        'lstm_pred'   : float(pred_l[-1]),
        'prophet_pred': float(pred_p[-1]),
        'neural_pred' : float(pred_np[-1]),
    }
    st.session_state['stock_chart'] = {
        'df_recent': df.tail(6 * 21),
        'dates_l'  : dates_l,
        'pred_l'   : pred_l,
        'dates_p'  : dates_p,
        'pred_p'   : pred_p,
        'dates_np' : dates_np,
        'pred_np'  : pred_np,
    }
    if 'stock_history' not in st.session_state:
        st.session_state['stock_history'] = []


# ============================================================
# AFFICHAGE GRAPHIQUE + MÉTRIQUES
# ============================================================

if 'stock_chart' in st.session_state:
    ctx   = st.session_state['stock_context']
    chart = st.session_state['stock_chart']

    df_r = chart['df_recent']
    fig  = go.Figure()

    fig.add_trace(go.Scatter(
        x    = df_r.index,
        y    = df_r['Close'].values.flatten(),
        name = 'Historique',
        line = dict(color='black', width=2)
    ))
    fig.add_trace(go.Scatter(
        x    = chart['dates_l'],
        y    = chart['pred_l'],
        name = 'LSTM',
        line = dict(color='steelblue',
                    width=1.5, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x    = chart['dates_p'],
        y    = chart['pred_p'],
        name = 'Prophet',
        line = dict(color='darkorange',
                    width=1.5, dash='dashdot')
    ))
    fig.add_trace(go.Scatter(
        x    = chart['dates_np'],
        y    = chart['pred_np'],
        name = 'NeuralProphet',
        line = dict(color='green',
                    width=1.5, dash='dot')
    ))
    fig.add_vline(
        x               = str(df_r.index[-1].date()),
        line_dash       = "dash",
        line_color      = "gray",
        annotation_text = "Aujourd'hui"
    )
    fig.update_layout(
        title       = (f"{ctx['company']} — "
                       f"{ctx['months']} mois"),
        xaxis_title = "Date",
        yaxis_title = "Prix ($)",
        height      = 500,
        hovermode   = "x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Métriques
    cp = ctx['current']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Prix actuel", f"${cp:.2f}")
    c2.metric("🤖 LSTM",
              f"${ctx['lstm_pred']:.2f}",
              f"{((ctx['lstm_pred']/cp)-1)*100:.1f}%")
    c3.metric("📊 Prophet",
              f"${ctx['prophet_pred']:.2f}",
              f"{((ctx['prophet_pred']/cp)-1)*100:.1f}%")
    c4.metric("🧠 NeuralProphet",
              f"${ctx['neural_pred']:.2f}",
              f"{((ctx['neural_pred']/cp)-1)*100:.1f}%")


# ============================================================
# CHATBOT BOURSIER
# ============================================================

st.markdown("---")
st.subheader("💬 Chatbot Financier")

if 'stock_context' not in st.session_state:
    st.warning("⚠️ Lancez d'abord une prédiction "
               "pour activer le chatbot.")
else:
    ctx = st.session_state['stock_context']
    st.info(
        f"📊 **{ctx['company']}** | "
        f"{ctx['months']} mois | "
        f"Prix actuel : **${ctx['current']:.2f}**"
    )

    if 'stock_history' not in st.session_state:
        st.session_state['stock_history'] = []

    # Afficher historique
    for msg in st.session_state['stock_history']:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # Input
    user_input = st.chat_input(
        "Posez une question sur cette action...")

    if user_input:
        st.session_state['stock_history'].append({
            "role"   : "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Réflexion..."):
            response = chat_stock(
                company      = ctx['company'],
                months       = ctx['months'],
                lstm_pred    = ctx['lstm_pred'],
                prophet_pred = ctx['prophet_pred'],
                neural_pred  = ctx['neural_pred'],
                conversation_history = \
                    st.session_state['stock_history']
            )

        st.session_state['stock_history'].append({
            "role"   : "assistant",
            "content": response
        })
        with st.chat_message("assistant"):
            st.write(response)
