# pages/2_Bourse.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import time
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from config import COMPANIES
from utils_stocks import (
    get_stock_data, get_close_value,
    load_lstm, predict_lstm,
    load_prophet_from_json, train_prophet, predict_prophet,
    check_neuralprophet, train_neuralprophet,
    predict_neuralprophet
)
from chatbot.llm_groq import chat_stock

if not hasattr(np, 'NaN'):
    np.NaN = np.nan

st.set_page_config(
    page_title = "Prédiction Boursière",
    page_icon  = "📈",
    layout     = "wide"
)

st.title("📈 Prédiction Boursière")
st.markdown("**LSTM · Prophet · NeuralProphet**")
st.markdown("---")


# ============================================================
# FALLBACK données simulées
# ============================================================

def get_fallback_data(company):
    np.random.seed(42)
    end    = datetime.now()
    start  = end - timedelta(days=730)
    dates  = pd.bdate_range(start=start, end=end)
    base   = {'NVIDIA': 178, 'ORACLE': 141,
               'IBM': 240, 'CISCO': 83}.get(company, 100)
    trend  = np.linspace(0, base * 0.1, len(dates))
    noise  = np.random.randn(len(dates)) * base * 0.02
    prices = (base + trend + noise).astype(float)
    df     = pd.DataFrame({'Close': prices}, index=dates)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# ============================================================
# CACHE
# ============================================================

@st.cache_data(ttl=3600)
def get_data_safe(company):
    try:
        df = get_stock_data(company)
        if df is not None and not df.empty \
           and 'Close' in df.columns and len(df) >= 30:
            _ = float(
                df['Close'].values.flatten().astype(float)[-1])
            return df
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance: {str(e)[:60]}")
    st.info(f"📊 Données simulées pour {company}")
    return get_fallback_data(company)


@st.cache_resource
def get_lstm_cached(company):
    try:
        return load_lstm(company)
    except Exception:
        return None


# ============================================================
# GRAPHIQUE — add_vline remplacé (cassé Python 3.14)
# ============================================================

def make_chart(df, predictions_dict, company, months):
    fig  = go.Figure()
    hist = df.tail(90)

    close_hist = hist['Close'].values.flatten().astype(float)
    fig.add_trace(go.Scatter(
        x    = hist.index,
        y    = close_hist,
        name = '📈 Historique',
        line = dict(color='#00ff88', width=2.5)
    ))

    colors = {
        'LSTM'         : '#00b4d8',
        'Prophet'      : '#ff6b35',
        'NeuralProphet': '#9b59b6'
    }
    dashes = {
        'LSTM'         : 'dash',
        'Prophet'      : 'dashdot',
        'NeuralProphet': 'dot'
    }

    for name, data in predictions_dict.items():
        fig.add_trace(go.Scatter(
            x    = data['dates'],
            y    = data['preds'].astype(float),
            name = f'🔮 {name}',
            line = dict(
                color = colors.get(name, '#fff'),
                width = 2,
                dash  = dashes.get(name, 'dash')
            )
        ))

    last_close = float(
        df['Close'].values.flatten().astype(float)[-1])
    today_str  = str(df.index[-1].date())

    fig.add_trace(go.Scatter(
        x      = [df.index[-1]],
        y      = [last_close],
        mode   = 'markers',
        name   = "📍 Aujourd'hui",
        marker = dict(size=14, color='#ff4444',
                      symbol='star')
    ))

    # ✅ Remplace add_vline (cassé Plotly/Python 3.14)
    # On utilise add_shape + add_annotation séparément
    fig.add_shape(
        type    = "line",
        x0      = today_str,
        x1      = today_str,
        y0      = 0,
        y1      = 1,
        yref    = "paper",
        line    = dict(color="gray", dash="dash", width=1)
    )
    fig.add_annotation(
        x         = today_str,
        y         = 1,
        yref      = "paper",
        text      = "Aujourd'hui",
        showarrow = False,
        font      = dict(color="gray", size=11),
        xanchor   = "left"
    )

    fig.update_layout(
        title        = f"📊 {company} — {months} mois",
        xaxis_title  = "Date",
        yaxis_title  = "Prix ($)",
        template     = "plotly_dark",
        height       = 500,
        hovermode    = "x unified",
        plot_bgcolor = 'rgba(0,0,0,0.2)',
        paper_bgcolor= 'rgba(0,0,0,0)',
        font         = dict(color='white'),
        legend       = dict(
            bgcolor = 'rgba(0,0,0,0.6)',
            font    = dict(color='white')
        )
    )
    return fig


# ============================================================
# SÉLECTEURS
# ============================================================

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox(
        "🏢 Entreprise", list(COMPANIES.keys()))
with col2:
    months = st.slider(
        "📅 Période (mois)",
        min_value=3, max_value=12, value=6)

c1, c2, c3 = st.columns(3)
with c1:
    use_lstm    = st.checkbox("✅ LSTM",    value=True)
with c2:
    use_prophet = st.checkbox("✅ Prophet", value=True)
with c3:
    use_neural  = st.checkbox(
        "🧠 NeuralProphet (entraîne en direct ~2 min)",
        value=False
    )

# Info NeuralProphet si coché
if use_neural:
    neural_ok = check_neuralprophet()
    if not neural_ok:
        st.error(
            "⚠️ **NeuralProphet incompatible avec ce serveur** "
            "(Python 3.14). Utilisez LSTM et Prophet."
        )
        use_neural = False
    else:
        st.warning(
            "🧠 **NeuralProphet sélectionné**\n\n"
            "Ce modèle s'entraîne **en direct** sur vos données "
            "(contrairement à LSTM et Prophet qui sont "
            "pré-chargés instantanément).\n\n"
            "⏱️ Durée estimée : **2 à 5 minutes**. "
            "Une fois entraîné, il sera mis en cache."
        )


# ============================================================
# BOUTON PRÉDICTION
# ============================================================

if st.button("🚀 Lancer les prédictions",
             type="primary",
             use_container_width=True):

    with st.spinner("📡 Chargement des données..."):
        df = get_data_safe(company)

    close_arr     = df['Close'].values.flatten().astype(float)
    current_price = float(close_arr[-1])

    st.success(
        f"✅ {len(df)} jours chargés | "
        f"Prix actuel : **${current_price:.2f}**"
    )

    predictions_dict = {}

    # --- LSTM ---
    if use_lstm:
        with st.spinner("🔮 LSTM en cours..."):
            try:
                model = get_lstm_cached(company)
                if model:
                    dates, preds = predict_lstm(
                        model, df, months)
                    predictions_dict['LSTM'] = {
                        'dates': dates,
                        'preds': np.array(preds).astype(float)
                    }
                    st.success("✅ LSTM terminé")
                else:
                    st.warning(
                        f"⚠️ `lstm_{company.lower()}.pt` "
                        f"introuvable à la racine du projet"
                    )
            except Exception as e:
                st.warning(f"⚠️ LSTM : {str(e)[:100]}")

    # --- Prophet ---
    if use_prophet:
        with st.spinner("📊 Prophet en cours..."):
            try:
                p_model = load_prophet_from_json(company)
                if p_model is None:
                    st.info(
                        "JSON Prophet non trouvé — "
                        "entraînement à la volée...")
                    p_model, _ = train_prophet(df)
                dates, preds = predict_prophet(
                    p_model, df, months)
                predictions_dict['Prophet'] = {
                    'dates': dates,
                    'preds': np.array(preds).astype(float)
                }
                st.success("✅ Prophet terminé")
            except Exception as e:
                st.warning(f"⚠️ Prophet : {str(e)[:100]}")

    # --- NeuralProphet ---
    if use_neural:
        st.markdown("---")
        st.markdown("#### 🧠 NeuralProphet — Entraînement en direct")

        neural_cached = (
            'np_model'   in st.session_state and
            st.session_state.get('np_company') == company
        )

        if neural_cached:
            st.success(
                f"⚡ Déjà entraîné pour {company} "
                f"— aucune attente !")
            np_m  = st.session_state['np_model']
            np_df = st.session_state['np_df']
        else:
            pb  = st.progress(0)
            box = st.empty()
            box.markdown(
                "🔄 **Étape 1/3** — Préparation des données...")
            pb.progress(15)
            time.sleep(0.3)
            box.markdown(
                "🔄 **Étape 2/3** — Chargement NeuralProphet...")
            pb.progress(35)
            time.sleep(0.3)
            box.markdown(
                "🔄 **Étape 3/3** — Entraînement réseau "
                "neuronal (20 epochs)... ☕")
            pb.progress(55)

            t0 = time.time()
            try:
                np_m, np_df = train_neuralprophet(df)
                elapsed     = time.time() - t0
                pb.progress(100)
                box.markdown(
                    f"✅ **Entraîné en {elapsed:.1f}s !**")
                st.session_state['np_model']   = np_m
                st.session_state['np_df']      = np_df
                st.session_state['np_company'] = company
                st.success("🎉 NeuralProphet prêt !")
            except Exception as e:
                st.error(f"❌ NeuralProphet : {e}")
                np_m = None

        if np_m is not None:
            try:
                dates, preds = predict_neuralprophet(
                    np_m, np_df, df, months)
                predictions_dict['NeuralProphet'] = {
                    'dates': dates,
                    'preds': np.array(preds).astype(float)
                }
                st.success("✅ NeuralProphet terminé")
            except Exception as e:
                st.warning(
                    f"⚠️ NeuralProphet predict : {str(e)[:100]}")

    # Sauvegarder et recharger
    if predictions_dict:
        st.session_state['stock_predictions']  = \
            predictions_dict
        st.session_state['stock_current_price'] = \
            current_price
        st.session_state['stock_company']       = company
        st.session_state['stock_months']        = months
        st.session_state['stock_df']            = df
        st.session_state['stock_chat_history']  = []
        st.rerun()
    else:
        st.error(
            "❌ Aucune prédiction générée.\n\n"
            "Vérifiez que `lstm_*.pt` et "
            "`prophet_*.json` sont bien à la racine "
            "du dépôt Streamlit Cloud."
        )


# ============================================================
# AFFICHAGE RÉSULTATS
# ============================================================

if 'stock_predictions' in st.session_state:
    predictions_dict = st.session_state['stock_predictions']
    current_price    = st.session_state['stock_current_price']
    company          = st.session_state['stock_company']
    months           = st.session_state['stock_months']
    df               = st.session_state.get('stock_df')

    if df is None:
        st.warning("Relancez une prédiction.")
        st.stop()

    # Courbes
    st.markdown("---")
    st.subheader("📈 Courbes des prédictions")
    fig = make_chart(df, predictions_dict, company, months)
    st.plotly_chart(fig, use_container_width=True)

    # Métriques Streamlit
    st.subheader("📊 Résultats")
    cols = st.columns(len(predictions_dict) + 1)
    cols[0].metric("💰 Prix actuel",
                   f"${current_price:.2f}")

    rows = []
    for i, (name, data) in enumerate(
            predictions_dict.items()):
        preds     = data['preds'].astype(float)
        final     = float(preds[-1])
        variation = (final - current_price) / \
                     current_price * 100

        if variation > 5:
            signal = "🟢 ACHAT FORT"
        elif variation > 2:
            signal = "🟡 ACHAT LÉGER"
        elif variation > -2:
            signal = "⚪ CONSERVER"
        elif variation > -5:
            signal = "🟠 VENDRE LÉGER"
        else:
            signal = "🔴 VENDRE FORT"

        cols[i + 1].metric(
            f"🔮 {name}",
            f"${final:.2f}",
            f"{variation:+.1f}%"
        )
        rows.append({
            "Modèle"      : name,
            "Prix prédit" : f"${final:.2f}",
            "Variation"   : f"{variation:+.1f}%",
            "Signal"      : signal
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )

    # Évolution mensuelle
    st.subheader("📅 Évolution mensuelle")
    evo_rows = []
    for name, data in predictions_dict.items():
        preds = data['preds'].astype(float)
        row   = {"Modèle": name}
        step  = 21
        for m in range(1, months + 1):
            idx        = min(m * step - 1, len(preds) - 1)
            row[f"M{m}"] = f"${preds[idx]:.2f}"
        evo_rows.append(row)

    st.dataframe(
        pd.DataFrame(evo_rows),
        use_container_width=True,
        hide_index=True
    )

    # Graphique barres
    st.subheader("📊 Comparaison finale")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x            = ["Prix actuel"],
        y            = [current_price],
        text         = f"${current_price:.2f}",
        textposition = 'auto',
        name         = "Prix actuel",
        marker_color = '#00ff88'
    ))
    colors_bar = {
        'LSTM': '#00b4d8', 'Prophet': '#ff6b35',
        'NeuralProphet': '#9b59b6'
    }
    for name, data in predictions_dict.items():
        final = float(data['preds'].astype(float)[-1])
        fig_bar.add_trace(go.Bar(
            x            = [name],
            y            = [final],
            text         = f"${final:.2f}",
            textposition = 'auto',
            name         = name,
            marker_color = colors_bar.get(name, '#fff')
        ))
    fig_bar.update_layout(
        title        = "Comparaison prix finaux prédits",
        yaxis_title  = "Prix ($)",
        template     = "plotly_dark",
        height       = 400,
        paper_bgcolor= 'rgba(0,0,0,0)',
        font         = dict(color='white')
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ============================================================
    # CHATBOT FINANCIER
    # ============================================================

    st.markdown("---")
    st.subheader("💬 Chatbot Financier")
    st.info(
        f"📊 **{company}** | {months} mois | "
        f"Prix actuel : **${current_price:.2f}**"
    )

    if 'stock_chat_history' not in st.session_state:
        st.session_state['stock_chat_history'] = []

    for msg in st.session_state['stock_chat_history']:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    user_input = st.chat_input(
        "Posez une question sur cette action...")

    if user_input:
        st.session_state['stock_chat_history'].append({
            "role": "user", "content": user_input
        })
        with st.chat_message("user"):
            st.write(user_input)

        lstm_pred    = float(predictions_dict['LSTM']['preds'].astype(float)[-1]) \
                       if 'LSTM' in predictions_dict else 0.0
        prophet_pred = float(predictions_dict['Prophet']['preds'].astype(float)[-1]) \
                       if 'Prophet' in predictions_dict else 0.0
        neural_pred  = float(predictions_dict['NeuralProphet']['preds'].astype(float)[-1]) \
                       if 'NeuralProphet' in predictions_dict else 0.0

        with st.spinner("Réflexion..."):
            response = chat_stock(
                company      = company,
                months       = months,
                lstm_pred    = lstm_pred,
                prophet_pred = prophet_pred,
                neural_pred  = neural_pred,
                conversation_history =
                    st.session_state['stock_chat_history']
            )

        st.session_state['stock_chat_history'].append({
            "role": "assistant", "content": response
        })
        with st.chat_message("assistant"):
            st.write(response)

st.markdown("---")
st.caption("⚠️ Prédictions IA — Pas un conseil financier")
