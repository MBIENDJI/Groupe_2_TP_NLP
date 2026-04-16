# pages/02_stock_prediction.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import json
import joblib
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Prédiction Boursière IA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS STYLE
# ============================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #00ff88 !important; }
    p, li, span, label, .stMarkdown { color: #ffffff !important; }
    [data-testid="stSidebar"] { background: #0a0a0a !important; border-right: 2px solid #00ff88 !important; }
    .stSelectbox > div > div { background-color: #1e1e2e !important; border: 2px solid #00ff88 !important; border-radius: 10px !important; }
    .stSelectbox > div > div > div { color: #ffffff !important; background-color: #1e1e2e !important; }
    div[data-baseweb="select"] ul { background-color: #1e1e2e !important; border: 1px solid #00ff88 !important; }
    div[data-baseweb="select"] li { color: #ffffff !important; background-color: #1e1e2e !important; }
    div[data-baseweb="select"] li:hover { background-color: #00ff88 !important; color: #000000 !important; }
    .stSlider > div > div > div { background-color: #00ff88 !important; }
    .stButton > button { background: linear-gradient(90deg, #00ff88, #00b4d8) !important; color: #000000 !important; font-weight: bold !important; border-radius: 30px !important; border: none !important; padding: 12px 24px !important; }
    .stButton > button:hover { transform: scale(1.02) !important; box-shadow: 0 0 20px rgba(0, 255, 136, 0.5) !important; }
    .stCheckbox label span { color: #ffffff !important; }
    .footer { text-align: center; padding: 20px; color: #666666; font-size: 0.8rem; border-top: 1px solid #00ff88; margin-top: 50px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================
COMPANIES = {
    'NVIDIA': 'NVDA',
    'ORACLE': 'ORCL',
    'IBM': 'IBM',
    'CISCO': 'CSCO'
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODÈLE LSTM
# ============================================================
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================
@st.cache_resource
def load_lstm(company):
    path = os.path.join(MODELS_DIR, f"lstm_{company.lower()}.pt")
    if os.path.exists(path):
        try:
            model = StockLSTM()
            checkpoint = torch.load(path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
            model = model.to(DEVICE)
            model.eval()
            return model
        except:
            return None
    return None

@st.cache_resource
def load_prophet(company):
    path = os.path.join(MODELS_DIR, f"prophet_{company.lower()}.json")
    if os.path.exists(path):
        try:
            from prophet import Prophet
            from prophet.serialize import model_from_json
            with open(path, 'r') as f:
                return model_from_json(json.load(f))
        except:
            return None
    return None

@st.cache_resource
def load_neuralprophet(company):
    path = os.path.join(MODELS_DIR, f"neuralprophet_{company.lower()}.pkl")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except:
            return None
    return None

# ============================================================
# DONNÉES
# ============================================================
@st.cache_data
def get_stock_data(company, period='2y'):
    try:
        stock = yf.Ticker(COMPANIES[company])
        df = stock.history(period=period)
        df.index = df.index.tz_localize(None)
        return df
    except:
        return None

# ============================================================
# FONCTION POUR CALCULER MAPE
# ============================================================
def calculate_mape(y_true, y_pred):
    """Calcule le Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Éviter division par zéro
    mask = y_true != 0
    if np.sum(mask) == 0:
        return float('inf')
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

# ============================================================
# PRÉDICTIONS
# ============================================================
def predict_lstm(model, df, days):
    from sklearn.preprocessing import MinMaxScaler
    
    prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(prices)
    
    window = 60
    if len(prices) < window:
        return np.array([prices[-1][0]] * days), float('inf')
    
    last = prices[-window:].flatten()
    last_scaled = scaler.transform(last.reshape(-1, 1)).flatten()
    
    preds = []
    current = last_scaled.copy()
    last_price = prices[-1][0]
    
    # Garder les vraies valeurs pour calculer MAPE sur validation
    y_true_validation = prices[-window:].flatten()
    
    model.eval()
    with torch.no_grad():
        for _ in range(days):
            x = torch.tensor(current, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            p = model(x).cpu().numpy()[0][0]
            preds.append(p)
            current = np.append(current[1:], p)
    
    preds = np.array(preds).reshape(-1, 1)
    preds = scaler.inverse_transform(preds).flatten()
    
    # Calculer MAPE sur les prédictions de validation
    y_pred_validation = scaler.inverse_transform(current[-window:].reshape(-1, 1)).flatten()
    mape = calculate_mape(y_true_validation[-len(y_pred_validation):], y_pred_validation)
    
    # Ajustement
    gap = last_price - preds[0]
    fade = np.linspace(1, 0, len(preds))
    preds = preds + gap * fade
    
    return preds, mape

def predict_prophet(model, df, days):
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')
    
    # Calculer MAPE sur données historiques
    historical = df['Close'].values[-60:]
    historical_dates = df.index[-60:]
    
    try:
        future = pd.DataFrame({'ds': dates})
        forecast = model.predict(future)
        preds = forecast['yhat'].values
        
        # Prédictions sur historique pour MAPE
        hist_forecast = model.predict(pd.DataFrame({'ds': historical_dates}))
        hist_preds = hist_forecast['yhat'].values
        mape = calculate_mape(historical, hist_preds)
    except:
        preds = np.array([last_price] * days)
        mape = float('inf')
    
    gap = last_price - preds[0]
    fade = np.linspace(1, 0, len(preds))
    preds = preds + gap * fade
    
    return preds, dates, mape

def predict_neuralprophet(model, df, days):
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')
    
    # Calculer MAPE sur données historiques
    historical = df['Close'].values[-60:]
    historical_dates = df.index[-60:]
    
    try:
        future = pd.DataFrame({'ds': dates})
        forecast = model.predict(future)
        col = 'yhat1' if 'yhat1' in forecast.columns else 'yhat'
        preds = forecast[col].values
        
        # Prédictions sur historique pour MAPE
        hist_future = pd.DataFrame({'ds': historical_dates})
        hist_forecast = model.predict(hist_future)
        hist_preds = hist_forecast[col].values
        mape = calculate_mape(historical, hist_preds)
    except:
        preds = np.array([last_price] * days)
        mape = float('inf')
    
    gap = last_price - preds[0]
    fade = np.linspace(1, 0, len(preds))
    preds = preds + gap * fade
    
    return preds, dates, mape

# ============================================================
# GRAPHIQUE
# ============================================================
def make_chart(df, predictions_dict, company, months):
    fig = go.Figure()
    
    hist_data = df.tail(90)
    hist_dates = hist_data.index.strftime('%Y-%m-%d').tolist()
    hist_prices = hist_data['Close'].values.tolist()
    
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices, mode='lines',
        name='📈 Historique', line=dict(color='#00ff88', width=2.5)
    ))
    
    colors = {'LSTM': '#00b4d8', 'Prophet': '#ff6b35', 'NeuralProphet': '#9b59b6'}
    
    for model_name, data in predictions_dict.items():
        preds = data['predictions']
        dates = data['dates']
        dates_str = [d.strftime('%Y-%m-%d') for d in dates]
        
        fig.add_trace(go.Scatter(
            x=dates_str, y=preds, mode='lines',
            name=f'🔮 {model_name} (MAPE: {data["mape"]:.1f}%)',
            line=dict(color=colors.get(model_name, '#ffffff'), width=2, dash='dash')
        ))
    
    today = df.index[-1]
    fig.add_trace(go.Scatter(
        x=[today.strftime('%Y-%m-%d')], y=[df['Close'].iloc[-1]],
        mode='markers', name='📍 Aujourd\'hui',
        marker=dict(size=14, color='#ff4444', symbol='star')
    ))
    
    fig.update_layout(
        title=f"📊 Prédictions {company} - {months} mois",
        xaxis_title="Date", yaxis_title="Prix ($)",
        template="plotly_dark", plot_bgcolor='rgba(0,0,0,0.2)',
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        height=500
    )
    return fig

# ============================================================
# INTERFACE PRINCIPALE
# ============================================================
def main():
    st.markdown("# 📈 Prédiction des Actions Boursières")
    st.markdown("### 🤖 Intelligence Artificielle - LSTM | Prophet | NeuralProphet")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("## 🎯 Configuration")
        
        company = st.selectbox(
            "🏢 Entreprise",
            options=list(COMPANIES.keys()),
            format_func=lambda x: f"{x} ({COMPANIES[x]})"
        )
        
        months = st.slider("📅 Période de prédiction", 3, 12, 6)
        
        st.markdown("---")
        st.markdown("### 🤖 Modèles à utiliser")
        
        use_lstm = st.checkbox("✅ LSTM (Deep Learning)", value=True)
        use_prophet = st.checkbox("✅ Prophet (Facebook)", value=True)
        use_neural = st.checkbox("✅ NeuralProphet (Uber)", value=True)
        
        st.markdown("---")
        st.markdown("### 📁 Modèles disponibles")
        
        lstm_path = os.path.join(MODELS_DIR, f"lstm_{company.lower()}.pt")
        prophet_path = os.path.join(MODELS_DIR, f"prophet_{company.lower()}.json")
        neural_path = os.path.join(MODELS_DIR, f"neuralprophet_{company.lower()}.pkl")
        
        if os.path.exists(lstm_path):
            st.success("LSTM ✅")
        else:
            st.error("LSTM ❌")
        
        if os.path.exists(prophet_path):
            st.success("Prophet ✅")
        else:
            st.error("Prophet ❌")
        
        if os.path.exists(neural_path):
            st.success("NeuralProphet ✅")
        else:
            st.error("NeuralProphet ❌")
        
        st.markdown("---")
        st.info("💡 **NeuralProphet** est plus lent à charger en raison de sa taille (~500MB)")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(f"🚀 PRÉDIRE {company} - {months} MOIS", use_container_width=True)
    
    if predict_button:
        with st.spinner(f"📊 Chargement des données {company}..."):
            df = get_stock_data(company)
            if df is None:
                st.error("Impossible de charger les données")
                return
            
            days = months * 21
            current_price = df['Close'].iloc[-1]
            st.success(f"✅ Données: {len(df)} jours | Prix actuel: ${current_price:.2f}")
        
        predictions_dict = {}
        
        # LSTM
        if use_lstm:
            with st.spinner("🔮 Prédiction LSTM..."):
                model = load_lstm(company)
                if model:
                    preds, mape = predict_lstm(model, df, days)
                    dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=days, freq='B')
                    predictions_dict['LSTM'] = {'predictions': preds, 'dates': dates, 'mape': mape}
                    st.success(f"✅ LSTM terminé (MAPE: {mape:.2f}%)")
                else:
                    st.warning("⚠️ Modèle LSTM non trouvé")
        
        # Prophet
        if use_prophet:
            with st.spinner("🔮 Prédiction Prophet..."):
                model = load_prophet(company)
                if model:
                    preds, dates, mape = predict_prophet(model, df, days)
                    predictions_dict['Prophet'] = {'predictions': preds, 'dates': dates, 'mape': mape}
                    st.success(f"✅ Prophet terminé (MAPE: {mape:.2f}%)")
                else:
                    st.warning("⚠️ Modèle Prophet non trouvé")
        
        # NeuralProphet - AVERTISSEMENT SUR LA LOURDEUR
        if use_neural:
            st.info("⏳ **NeuralProphet** est en cours de chargement... Modèle volumineux (~500MB), cela peut prendre 30-60 secondes.")
            with st.spinner("🔮 Prédiction NeuralProphet (modèle lourd en cours...)"):
                model = load_neuralprophet(company)
                if model:
                    preds, dates, mape = predict_neuralprophet(model, df, days)
                    predictions_dict['NeuralProphet'] = {'predictions': preds, 'dates': dates, 'mape': mape}
                    st.success(f"✅ NeuralProphet terminé (MAPE: {mape:.2f}%)")
                else:
                    st.warning("⚠️ Modèle NeuralProphet non trouvé")
        
        if predictions_dict:
            st.markdown("---")
            fig = make_chart(df, predictions_dict, company, months)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau avec MAPE
            st.subheader("📊 Comparaison des modèles")
            results = []
            for model_name, data in predictions_dict.items():
                preds = data['predictions']
                final_price = preds[-1]
                variation = (final_price - current_price) / current_price * 100
                mape = data['mape']
                
                if variation > 5:
                    rec = "🟢 ACHAT FORT"
                elif variation > 2:
                    rec = "🟡 ACHAT"
                elif variation > -2:
                    rec = "⚪ CONSERVER"
                elif variation > -5:
                    rec = "🟠 VENDRE"
                else:
                    rec = "🔴 VENDRE FORT"
                
                results.append({
                    "Modèle": model_name,
                    "MAPE": f"{mape:.2f}%" if mape != float('inf') else "N/A",
                    "Prix actuel": f"${current_price:.2f}",
                    "Prix prédit": f"${final_price:.2f}",
                    "Variation": f"{variation:+.2f}%",
                    "Recommandation": rec
                })
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            # Meilleur modèle selon MAPE
            best_model = min(results, key=lambda x: float(x['MAPE'].replace('%', '')) if x['MAPE'] != 'N/A' else float('inf'))
            st.success(f"🏆 **Meilleur modèle selon MAPE:** {best_model['Modèle']} (MAPE: {best_model['MAPE']})")
        else:
            st.error("❌ Aucune prédiction générée")
    
    st.markdown("""
    <div class="footer">
        ⚠️ <strong>Avertissement :</strong> Les prédictions sont générées par des modèles IA.<br>
        Ces informations ne constituent pas un conseil financier.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
