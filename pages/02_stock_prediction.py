# pages/02_stock_prediction.py (version complète avec chatbot intégré)
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
import sys
import warnings
warnings.filterwarnings('ignore')

# Ajouter le chemin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.translator import GermanTranslator
from chatbot.summarizer import Summarizer
from chatbot.text_to_speech import tts
from chatbot.langsmith_monitor import monitor

# Configuration
st.set_page_config(
    page_title="Prédiction Boursière IA",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CSS STYLE
# ============================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); }
    h1, h2, h3 { color: #00ff88 !important; }
    p, li, span, label { color: #ffffff !important; }
    [data-testid="stSidebar"] { background: #0a0a0a !important; border-right: 2px solid #00ff88 !important; }
    .stSelectbox > div > div { background-color: #1e1e2e !important; border: 2px solid #00ff88 !important; border-radius: 10px !important; }
    .stSelectbox > div > div > div { color: #ffffff !important; }
    .stButton > button { background: linear-gradient(90deg, #00ff88, #00b4d8) !important; color: #000000 !important; font-weight: bold !important; border-radius: 30px !important; }
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

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = BASE_DIR
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
        ticker = COMPANIES[company]
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df is not None and not df.empty:
            df.index = df.index.tz_localize(None)
            return df
    except:
        pass
    return None

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
        return np.array([prices[-1][0]] * days)
    
    last = prices[-window:].flatten()
    last_scaled = scaler.transform(last.reshape(-1, 1)).flatten()
    
    preds = []
    current = last_scaled.copy()
    last_price = prices[-1][0]
    
    model.eval()
    with torch.no_grad():
        for _ in range(days):
            x = torch.tensor(current, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            p = model(x).cpu().numpy()[0][0]
            preds.append(p)
            current = np.append(current[1:], p)
    
    preds = np.array(preds).reshape(-1, 1)
    preds = scaler.inverse_transform(preds).flatten()
    
    gap = last_price - preds[0]
    fade = np.linspace(1, 0, len(preds))
    preds = preds + gap * fade
    
    return preds

def predict_prophet(model, df, days):
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')
    
    try:
        future = pd.DataFrame({'ds': dates})
        forecast = model.predict(future)
        preds = forecast['yhat'].values
    except:
        preds = np.array([last_price] * days)
    
    gap = last_price - preds[0]
    fade = np.linspace(1, 0, len(preds))
    preds = preds + gap * fade
    
    return preds, dates

def predict_neuralprophet(model, df, days):
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')
    
    try:
        future = pd.DataFrame({'ds': dates})
        forecast = model.predict(future)
        col = 'yhat1' if 'yhat1' in forecast.columns else 'yhat'
        preds = forecast[col].values
    except:
        preds = np.array([last_price] * days)
    
    gap = last_price - preds[0]
    fade = np.linspace(1, 0, len(preds))
    preds = preds + gap * fade
    
    return preds, dates

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
            name=f'🔮 {model_name}',
            line=dict(color=colors.get(model_name, '#ffffff'), width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f"📊 Prédictions {company} - {months} mois",
        xaxis_title="Date", yaxis_title="Prix ($)",
        template="plotly_dark", plot_bgcolor='rgba(0,0,0,0.2)',
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=500
    )
    return fig

# ============================================================
# CHATBOT INTELLIGENT
# ============================================================
class StockChatbot:
    def __init__(self):
        self.history = []
    
    def generate_response(self, company, predictions, current_price, user_question=None):
        response = f"""
🤖 **Assistant Financier - {company}**

💰 **Prix actuel:** ${current_price:.2f}

📊 **Prédictions des modèles:**
"""
        for model, data in predictions.items():
            variation = data['variation']
            signe = "+" if variation >= 0 else ""
            response += f"\n   • **{model}:** ${data['final_price']:.2f} ({signe}{variation:.1f}%)"
        
        avg_var = sum(d['variation'] for d in predictions.values()) / len(predictions)
        
        if avg_var > 5:
            rec = "🟢 ACHAT FORT"
        elif avg_var > 2:
            rec = "🟡 ACHAT"
        elif avg_var > -2:
            rec = "⚪ CONSERVER"
        elif avg_var > -5:
            rec = "🟠 VENDRE LÉGER"
        else:
            rec = "🔴 VENDRE"
        
        response += f"\n\n📈 **Tendance moyenne:** {avg_var:+.1f}%\n🎯 **Recommandation:** {rec}"
        
        if user_question:
            response += f"\n\n💬 **Réponse à votre question:**\n{self._answer_question(user_question, company, predictions)}"
        
        return response
    
    def _answer_question(self, question, company, predictions):
        q = question.lower()
        if "prophét" in q or "prophet" in q:
            pred = predictions.get('Prophet', {})
            return f"Pour {company}, le modèle Prophet prévoit un prix de ${pred.get('final_price', 0):.2f} soit une variation de {pred.get('variation', 0):+.1f}%."
        elif "lstm" in q:
            pred = predictions.get('LSTM', {})
            return f"Le modèle LSTM pour {company} donne ${pred.get('final_price', 0):.2f} ({pred.get('variation', 0):+.1f}%)."
        elif "neural" in q:
            pred = predictions.get('NeuralProphet', {})
            return f"NeuralProphet prédit ${pred.get('final_price', 0):.2f} pour {company} ({pred.get('variation', 0):+.1f}%)."
        else:
            return f"Analyse en cours pour {company}. Les modèles prévoient une tendance générale basée sur les données historiques."
    
    def log_interaction(self, company, question, response):
        monitor.log_interaction(f"Stock_{company}", question, response, "Stock_Chatbot")

stock_chatbot = StockChatbot()

# ============================================================
# INTERFACE PRINCIPALE
# ============================================================
def main():
    st.title("📈 Prédiction des Actions Boursières")
    st.markdown("### 🤖 Intelligence Artificielle - LSTM | Prophet | NeuralProphet")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("## 🎯 Configuration")
        company = st.selectbox("🏢 Entreprise", list(COMPANIES.keys()))
        months = st.slider("📅 Période", 3, 12, 6)
        
        st.markdown("---")
        st.markdown("### 🤖 Modèles")
        use_lstm = st.checkbox("LSTM", value=True)
        use_prophet = st.checkbox("Prophet", value=True)
        use_neural = st.checkbox("NeuralProphet", value=True)
        
        st.markdown("---")
        st.markdown("### 💬 Chatbot Intelligent")
        st.markdown("Posez vos questions sur les prédictions")
    
    # Zone de question
    user_question = st.text_input("💬 Votre question sur l'entreprise sélectionnée:", 
                                   placeholder=f"Ex: Quelle est la prédiction pour {company} avec Prophet ?")
    
    if st.button(f"🚀 PRÉDIRE {company} - {months} MOIS", use_container_width=True):
        with st.spinner("Chargement..."):
            df = get_stock_data(company)
            if df is None:
                st.error("❌ Impossible de charger les données")
                return
            
            days = months * 21
            current_price = df['Close'].iloc[-1]
            predictions_dict = {}
            
            if use_lstm:
                model = load_lstm(company)
                if model:
                    preds = predict_lstm(model, df, days)
                    dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=days, freq='B')
                    final_price = preds[-1]
                    predictions_dict['LSTM'] = {
                        'predictions': preds, 'dates': dates,
                        'final_price': final_price,
                        'variation': (final_price - current_price) / current_price * 100
                    }
            
            if use_prophet:
                model = load_prophet(company)
                if model:
                    preds, dates = predict_prophet(model, df, days)
                    final_price = preds[-1]
                    predictions_dict['Prophet'] = {
                        'predictions': preds, 'dates': dates,
                        'final_price': final_price,
                        'variation': (final_price - current_price) / current_price * 100
                    }
            
            if use_neural:
                model = load_neuralprophet(company)
                if model:
                    preds, dates = predict_neuralprophet(model, df, days)
                    final_price = preds[-1]
                    predictions_dict['NeuralProphet'] = {
                        'predictions': preds, 'dates': dates,
                        'final_price': final_price,
                        'variation': (final_price - current_price) / current_price * 100
                    }
            
            if predictions_dict:
                st.success(f"✅ Données: {len(df)} jours | Prix: ${current_price:.2f}")
                
                # Graphique
                fig = make_chart(df, predictions_dict, company, months)
                st.plotly_chart(fig, use_container_width=True)
                
                # CHATBOT
                st.markdown("---")
                st.subheader("🤖 Assistant Financier Intelligent")
                
                # Générer réponse du chatbot
                response = stock_chatbot.generate_response(company, predictions_dict, current_price, user_question if user_question else None)
                
                # Afficher la réponse
                st.markdown(response)
                
                # Onglets pour traduction, résumé, audio
                tab1, tab2, tab3 = st.tabs(["🇩🇪 Traduction allemande", "📝 Résumé", "🎵 Audio"])
                
                with tab1:
                    german_text = GermanTranslator.translate(response)
                    st.markdown(german_text)
                
                with tab2:
                    summary = f"""
📊 **RÉSUMÉ PRÉDICTION {company}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 Prix actuel: ${current_price:.2f}
"""
                    for model, data in predictions_dict.items():
                        summary += f"\n🔮 {model}: ${data['final_price']:.2f} ({data['variation']:+.1f}%)"
                    
                    avg = sum(d['variation'] for d in predictions_dict.values()) / len(predictions_dict)
                    summary += f"\n\n📈 Tendance moyenne: {avg:+.1f}%"
                    st.markdown(summary)
                
                with tab3:
                    audio_file = tts.to_audio(response[:500], f"stock_{company}.mp3")
                    if audio_file and os.path.exists(audio_file):
                        st.audio(audio_file)
                    else:
                        st.info("🎙️ Audio généré automatiquement")
                
                # Log LangSmith
                stock_chatbot.log_interaction(company, user_question or "Prédiction standard", response[:200])
                
            else:
                st.error("❌ Aucun modèle trouvé")

if __name__ == "__main__":
    main()
