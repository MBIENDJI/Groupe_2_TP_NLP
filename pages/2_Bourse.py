# pages/2_Bourse.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os, time
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from config       import COMPANIES
from utils_stocks import (get_stock_data, load_lstm,
                           predict_lstm,
                           load_prophet_from_json,
                           train_prophet,
                           predict_prophet,
                           check_neuralprophet,
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
    "Prédictions **LSTM** · **Prophet** · **NeuralProphet**")
st.markdown("---")


# ============================================================
# HELPERS MÉTRIQUES
# ============================================================

def compute_mape(actual, predicted):
    """MAPE sur les N derniers jours comparables."""
    actual    = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    n         = min(len(actual), len(predicted))
    actual    = actual[-n:]
    predicted = predicted[-n:]
    mask      = actual != 0
    return float(np.mean(np.abs(
        (actual[mask] - predicted[mask]) / actual[mask]
    )) * 100)


def compute_rmse(actual, predicted):
    actual    = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    n         = min(len(actual), len(predicted))
    return float(np.sqrt(np.mean(
        (actual[-n:] - predicted[-n:]) ** 2
    )))


# ============================================================
# GRAPHIQUE PLOTLY COMPLET
# ============================================================

def make_chart(df, predictions_dict, company, months):
    fig = go.Figure()

    # Historique 90 derniers jours
    hist = df.tail(90)
    fig.add_trace(go.Scatter(
        x    = hist.index,
        y    = hist['Close'].values.flatten(),
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
            y    = data['preds'],
            name = f'🔮 {name}',
            line = dict(
                color = colors.get(name, '#fff'),
                width = 2,
                dash  = dashes.get(name, 'dash')
            )
        ))

    # Marqueur aujourd'hui
    fig.add_trace(go.Scatter(
        x      = [df.index[-1]],
        y      = [float(df['Close'].iloc[-1])],
        mode   = 'markers',
        name   = "📍 Aujourd'hui",
        marker = dict(size=14, color='#ff4444',
                      symbol='star')
    ))

    fig.update_layout(
        title       = dict(
            text = f"📊 {company} — {months} mois",
            font = dict(size=20, color='#00ff88'), x=0.5
        ),
        xaxis_title  = "Date",
        yaxis_title  = "Prix ($)",
        template     = "plotly_dark",
        plot_bgcolor = 'rgba(0,0,0,0.2)',
        paper_bgcolor= 'rgba(0,0,0,0)',
        font         = dict(color='white'),
        hovermode    = 'x unified',
        height       = 500,
        legend       = dict(
            bgcolor = 'rgba(0,0,0,0.6)',
            font    = dict(color='white')
        )
    )
    return fig


def make_bar_chart(predictions_dict, current_price):
    fig = go.Figure()
    colors = {
        'LSTM': '#00b4d8', 'Prophet': '#ff6b35',
        'NeuralProphet': '#9b59b6'
    }
    for name, data in predictions_dict.items():
        final = float(data['preds'][-1])
        fig.add_trace(go.Bar(
            x             = [name],
            y             = [final],
            name          = name,
            text          = f"${final:.2f}",
            textposition  = 'auto',
            marker_color  = colors.get(name, '#888')
        ))
    fig.add_trace(go.Bar(
        x            = ['Prix actuel'],
        y            = [current_price],
        name         = 'Prix actuel',
        text         = f"${current_price:.2f}",
        textposition = 'auto',
        marker_color = '#00ff88'
    ))
    fig.update_layout(
        title        = "Comparaison des prix finaux prédits",
        yaxis_title  = "Prix ($)",
        template     = "plotly_dark",
        paper_bgcolor= 'rgba(0,0,0,0)',
        font         = dict(color='white'),
        showlegend   = False
    )
    return fig


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
        min_value=3, max_value=12, value=6
    )

# Sélecteur modèles
st.markdown("#### 🤖 Modèles à utiliser")
c1, c2, c3 = st.columns(3)
with c1:
    use_lstm    = st.checkbox("✅ LSTM",          value=True)
with c2:
    use_prophet = st.checkbox("✅ Prophet",       value=True)
with c3:
    use_neural  = st.checkbox(
        "🧠 NeuralProphet (entraînement en direct ~2-5 min)",
        value=False
    )

# Vérifier disponibilité NeuralProphet côté serveur
neural_ok = check_neuralprophet()
if use_neural and not neural_ok:
    st.error(
        "⚠️ **NeuralProphet est incompatible avec Python "
        f"{__import__('sys').version_info.major}."
        f"{__import__('sys').version_info.minor} "
        "sur ce serveur.**\n\n"
        "Utilisez LSTM et Prophet qui fonctionnent parfaitement."
    )
    use_neural = False

# ============================================================
# CHARGEMENT DONNÉES ET MODÈLES (cache)
# ============================================================

@st.cache_resource
def get_lstm_cached(name):
    return load_lstm(name)

@st.cache_data
def get_data_cached(name):
    return get_stock_data(name)


# ============================================================
# BOUTON PRÉDICTION
# ============================================================

if st.button("🚀 Lancer les prédictions", type="primary"):

    with st.spinner("📡 Téléchargement des données..."):
        df = get_data_cached(company)

    current_price = float(df['Close'].iloc[-1])
    st.success(
        f"✅ {len(df)} jours chargés | "
        f"Prix actuel : **${current_price:.2f}**"
    )

    predictions_dict = {}

    # ----------------------------------------------------------
    # LSTM
    # ----------------------------------------------------------
    if use_lstm:
        with st.spinner("🔮 Prédiction LSTM..."):
            try:
                lstm_model     = get_lstm_cached(company)
                dates_l, pred_l = predict_lstm(
                    lstm_model, df, months)
                predictions_dict['LSTM'] = {
                    'dates': dates_l, 'preds': pred_l
                }
                st.success("✅ LSTM terminé")
            except Exception as e:
                st.warning(f"⚠️ LSTM échoué : {e}")

    # ----------------------------------------------------------
    # PROPHET
    # ----------------------------------------------------------
    if use_prophet:
        with st.spinner("📊 Prédiction Prophet..."):
            try:
                p_model = load_prophet_from_json(company)
                if p_model is None:
                    st.info(
                        "Modèle Prophet JSON non trouvé "
                        "— entraînement à la volée..."
                    )
                    p_model, _ = train_prophet(df)
                dates_p, pred_p = predict_prophet(
                    p_model, df, months)
                predictions_dict['Prophet'] = {
                    'dates': dates_p, 'preds': pred_p
                }
                st.success("✅ Prophet terminé")
            except Exception as e:
                st.warning(f"⚠️ Prophet échoué : {e}")

    # ----------------------------------------------------------
    # NEURALPROPHET — message patience + entraînement en direct
    # ----------------------------------------------------------
    if use_neural:
        st.markdown("---")
        st.markdown("### 🧠 NeuralProphet — Entraînement en direct")
        st.warning(
            "**Pourquoi cette attente ?**\n\n"
            "Contrairement à LSTM et Prophet qui sont "
            "**pré-chargés instantanément**, NeuralProphet "
            "s'entraîne **en temps réel sur vos données**.\n\n"
            "| Modèle | Mode | Durée |\n"
            "|--------|------|-------|\n"
            "| LSTM | Pré-entraîné ✅ | < 1 sec |\n"
            "| Prophet | Pré-entraîné ✅ | < 1 sec |\n"
            "| **NeuralProphet** | **En direct 🔄** | **2–5 min** |\n\n"
            "☕ Le réseau neuronal apprend les tendances "
            "sur 5 ans de données historiques (20 epochs). "
            "Résultat mis en cache ensuite."
        )

        neural_cached = (
            'neural_model'   in st.session_state and
            st.session_state.get('neural_company') == company
        )

        if neural_cached:
            st.success(
                f"⚡ NeuralProphet déjà entraîné pour "
                f"{company} — aucune attente !"
            )
            np_model = st.session_state['neural_model']
            np_df    = st.session_state['neural_df']
        else:
            pb  = st.progress(0)
            box = st.empty()
            box.markdown("🔄 **Étape 1/3** — Préparation des données...")
            pb.progress(20)
            time.sleep(0.3)
            box.markdown("🔄 **Étape 2/3** — Chargement du module NeuralProphet...")
            pb.progress(40)
            t0 = time.time()
            try:
                np_model, np_df = train_neuralprophet(df)
                elapsed = time.time() - t0
                pb.progress(100)
                box.markdown(
                    f"✅ **Entraîné en {elapsed:.1f}s !**"
                )
                st.session_state['neural_model']   = np_model
                st.session_state['neural_df']      = np_df
                st.session_state['neural_company'] = company
                st.success(
                    f"🎉 NeuralProphet prêt et mis en cache !"
                )
            except RuntimeError as e:
                st.error(str(e))
                np_model = None

        if np_model is not None:
            try:
                dates_np, pred_np = predict_neuralprophet(
                    np_model, np_df, df, months)
                predictions_dict['NeuralProphet'] = {
                    'dates': dates_np, 'preds': pred_np
                }
                st.success("✅ NeuralProphet terminé")
            except Exception as e:
                st.warning(f"⚠️ NeuralProphet predict : {e}")

    # ----------------------------------------------------------
    # Stocker résultats en session pour chatbot
    # ----------------------------------------------------------
    if predictions_dict:
        st.session_state['stock_context'] = {
            'company'     : company,
            'months'      : months,
            'current'     : current_price,
            'lstm_pred'   : float(predictions_dict['LSTM']['preds'][-1])
                            if 'LSTM' in predictions_dict else 0,
            'prophet_pred': float(predictions_dict['Prophet']['preds'][-1])
                            if 'Prophet' in predictions_dict else 0,
            'neural_pred' : float(predictions_dict['NeuralProphet']['preds'][-1])
                            if 'NeuralProphet' in predictions_dict else 0,
        }
        st.session_state['stock_preds'] = predictions_dict
        if 'stock_history' not in st.session_state:
            st.session_state['stock_history'] = []


# ============================================================
# AFFICHAGE RÉSULTATS
# ============================================================

if 'stock_preds' in st.session_state:
    ctx              = st.session_state['stock_context']
    predictions_dict = st.session_state['stock_preds']
    current_price    = ctx['current']
    company_disp     = ctx['company']
    months_disp      = ctx['months']

    df_cached = get_data_cached(company_disp)

    st.markdown("---")
    st.subheader("📊 Courbes de prédiction")
    fig = make_chart(
        df_cached, predictions_dict, company_disp, months_disp)
    st.plotly_chart(fig, use_container_width=True)

    # Graphique barres
    st.subheader("📊 Comparaison des prix finaux")
    fig_bar = make_bar_chart(predictions_dict, current_price)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ----------------------------------------------------------
    # MÉTRIQUES : prix + variation + MAPE + RMSE
    # ----------------------------------------------------------
    st.subheader("📐 Métriques de performance")

    actual_last60 = df_cached['Close'].values.flatten()[-60:]
    rows = []
    for name, data in predictions_dict.items():
        preds  = data['preds']
        final  = float(preds[-1])
        var    = (final - current_price) / current_price * 100
        mape   = compute_mape(actual_last60, preds[:60])
        rmse   = compute_rmse(actual_last60, preds[:60])

        if var > 5:
            rec = "🟢 ACHAT FORT"
        elif var > 2:
            rec = "🟡 ACHAT LÉGER"
        elif var > -2:
            rec = "⚪ CONSERVER"
        elif var > -5:
            rec = "🟠 VENDRE LÉGER"
        else:
            rec = "🔴 VENDRE FORT"

        rows.append({
            "Modèle"       : name,
            "Prix actuel"  : f"${current_price:.2f}",
            "Prix prédit"  : f"${final:.2f}",
            "Variation"    : f"{var:+.2f}%",
            "MAPE (%)"     : f"{mape:.2f}",
            "RMSE ($)"     : f"{rmse:.2f}",
            "Signal"       : rec
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )

    # Métriques Streamlit
    cols = st.columns(len(predictions_dict) + 1)
    cols[0].metric("💵 Prix actuel", f"${current_price:.2f}")
    for i, (name, data) in enumerate(predictions_dict.items()):
        final = float(data['preds'][-1])
        delta = ((final / current_price) - 1) * 100
        cols[i + 1].metric(
            f"🔮 {name}",
            f"${final:.2f}",
            f"{delta:+.1f}%"
        )

    # ----------------------------------------------------------
    # Évolution mensuelle
    # ----------------------------------------------------------
    with st.expander("📅 Évolution mensuelle détaillée"):
        evo_rows = []
        for name, data in predictions_dict.items():
            preds = data['preds']
            dates = data['dates']
            row   = {"Modèle": name}
            step  = 21  # 1 mois ≈ 21 jours ouvrables
            for m in range(0, len(preds), step):
                label     = f"Mois {m // step + 1}"
                row[label] = f"${preds[min(m, len(preds)-1)]:.2f}"
            evo_rows.append(row)
        st.dataframe(
            pd.DataFrame(evo_rows),
            use_container_width=True,
            hide_index=True
        )

    # ----------------------------------------------------------
    # CHATBOT BOURSIER — séparé des courbes
    # ----------------------------------------------------------
    st.markdown("---")
    st.subheader("💬 Chatbot Financier")
    st.info(
        f"📊 **{ctx['company']}** | "
        f"{ctx['months']} mois | "
        f"Prix actuel : **${ctx['current']:.2f}**"
    )

    if 'stock_history' not in st.session_state:
        st.session_state['stock_history'] = []

    for msg in st.session_state['stock_history']:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    user_input = st.chat_input(
        "Posez une question sur cette action...")

    if user_input:
        st.session_state['stock_history'].append({
            "role": "user", "content": user_input
        })
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("🤔 Réflexion..."):
            response = chat_stock(
                company      = ctx['company'],
                months       = ctx['months'],
                lstm_pred    = ctx['lstm_pred'],
                prophet_pred = ctx['prophet_pred'],
                neural_pred  = ctx['neural_pred'],
                conversation_history =
                    st.session_state['stock_history']
            )

        st.session_state['stock_history'].append({
            "role": "assistant", "content": response
        })
        with st.chat_message("assistant"):
            st.write(response)

else:
    st.info(
        "👆 Sélectionnez une entreprise, une période "
        "et les modèles souhaités, puis cliquez sur "
        "**Lancer les prédictions**."
    )

st.markdown("---")
st.caption(
    "⚠️ Ces prédictions sont générées par des modèles IA. "
    "Elles ne constituent pas un conseil financier."
)
