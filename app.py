# app.py
import streamlit as st

st.set_page_config(
    page_title = "Medical & Stock AI",
    page_icon  = "🤖",
    layout     = "wide"
)

st.title("🤖 Assistant Médical & Financier IA")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🏥 Détection Maladies Rénales
    - Classification CNN EfficientNet-B0
    - Détection : Normal, Kyste, Tumeur, Calcul
    - Chatbot médical intelligent
    - Traduction allemand + Résumé + Audio
    """)
    if st.button("→ Aller à la détection rénale",
                 type="primary",
                 use_container_width=True):
        st.switch_page("pages/1_Reins.py")

with col2:
    st.markdown("""
    ### 📈 Prédiction Boursière
    - LSTM + Prophet + NeuralProphet
    - Sélecteur période 3 à 12 mois
    - NVIDIA, ORACLE, IBM, CISCO
    - Chatbot financier intelligent
    """)
    if st.button("→ Aller aux prédictions boursières",
                 type="primary",
                 use_container_width=True):
        st.switch_page("pages/2_Bourse.py")

st.markdown("---")
st.caption(
    "Technologies : CNN EfficientNet-B0 | LSTM PyTorch | "
    "Prophet | NeuralProphet | Mistral-7B | LangSmith"
)
