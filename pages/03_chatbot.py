# pages/03_chatbot.py
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.stock_chatbot import stock_chatbot
from chatbot.translator import GermanTranslator
from chatbot.summarizer import Summarizer
from chatbot.text_to_speech import tts

st.set_page_config(
    page_title="Chatbot Financier",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Assistant Financier Intelligent")
st.markdown("### Posez vos questions sur les actions NVIDIA, ORACLE, IBM, CISCO")
st.markdown("---")

# Import des données depuis la session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'current_price' not in st.session_state:
    st.session_state.current_price = None

# Sidebar
with st.sidebar:
    st.header("🎯 Configuration")
    
    company = st.selectbox(
        "🏢 Choisissez une entreprise",
        ['NVIDIA', 'ORACLE', 'IBM', 'CISCO'],
        format_func=lambda x: f"{x}"
    )
    
    st.markdown("---")
    st.header("💡 Questions suggérées")
    st.markdown("""
    - Quelle est la tendance ?
    - Dois-je acheter ?
    - Quel est le risque ?
    - Comparé aux autres ?
    """)

# Interface principale
st.subheader(f"💬 Chatbot - {company}")

# Zone de chat
user_query = st.text_input("💬 Votre question:", 
                           placeholder=f"Ex: Quelle est la prédiction pour {company}?")

# Bouton pour générer une prédiction
if st.button(f"🔮 Analyser {company}", use_container_width=True):
    with st.spinner(f"Analyse de {company}..."):
        # Simulation de prédictions
        import numpy as np
        current_price = 100 + np.random.randn() * 50
        st.session_state.current_price = current_price
        
        # Simuler des prédictions
        predictions = {
            'LSTM': {'final_price': current_price * (1 + np.random.randn() * 0.1), 'variation': np.random.randn() * 10},
            'Prophet': {'final_price': current_price * (1 + np.random.randn() * 0.1), 'variation': np.random.randn() * 10},
            'NeuralProphet': {'final_price': current_price * (1 + np.random.randn() * 0.1), 'variation': np.random.randn() * 10}
        }
        st.session_state.predictions = predictions
        
        # Traiter avec le chatbot
        result = stock_chatbot.process_query(company, predictions, current_price)
        stock_chatbot.display_chatbot_ui(result)
        
        # Traduction en allemand
        st.markdown("---")
        st.subheader("🇩🇪 Traduction allemande")
        german_text = GermanTranslator.translate(result['response'])
        st.markdown(german_text)
        
        # Résumé
        st.subheader("📝 Résumé")
        summary = Summarizer.summarize_stock_prediction(company, predictions, current_price)
        st.markdown(summary)
        
        # Audio
        st.subheader("🎵 Version audio")
        audio_file = tts.to_audio(result['response'][:500], f"stock_{company}.mp3")
        if audio_file and os.path.exists(audio_file):
            st.audio(audio_file)

# Historique
with st.expander("📜 Historique des conversations"):
    st.write("Les conversations sont monitorées avec LangSmith")
    st.caption("🔍 Monitoring actif - Toutes les interactions sont enregistrées")
