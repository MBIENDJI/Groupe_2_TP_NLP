# pages/1_Reins.py
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from gtts import gTTS
import tempfile

from utils_cnn import load_cnn_model, predict_kidney
from chatbot.llm_groq import chat_kidney, translate_to_german, generate_summary

st.set_page_config(
    page_title="Détection Rénale",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Détection Maladies Rénales")
st.markdown("Classification par **CNN EfficientNet-B0**")
st.markdown("---")

CONFIDENCE_THRESHOLD = 0.70


@st.cache_resource
def get_cnn():
    return load_cnn_model("best_model.pth")

try:
    cnn_model = get_cnn()
    st.success("✅ Modèle CNN chargé")
except Exception as e:
    st.error(f"❌ Erreur chargement CNN : {e}")
    st.stop()


if st.button("🔄 Recommencer une nouvelle analyse"):
    for key in ['kidney_disease_fr', 'kidney_confidence', 'kidney_response',
                'kidney_history', 'kidney_summary', 'kidney_german', 'kidney_rejected']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


uploaded = st.file_uploader(
    "📁 Charger une image CT scan du rein",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Image CT chargée", use_container_width=True)

    with col2:
        if st.button("🔬 Analyser l'image", type="primary"):
            with st.spinner("Analyse CNN en cours..."):
                en, fr, conf, probs = predict_kidney(cnn_model, image)

            if conf < CONFIDENCE_THRESHOLD:
                st.session_state['kidney_rejected'] = True
                for key in ['kidney_disease_fr', 'kidney_confidence', 'kidney_response',
                            'kidney_history', 'kidney_summary', 'kidney_german']:
                    if key in st.session_state:
                        del st.session_state[key]

                st.error(f"⚠️ **Image non reconnue**\n\nConfiance : {conf:.1%} (seuil: {CONFIDENCE_THRESHOLD:.0%})")
                st.markdown("**Probabilités :**")
                for cls, prob in probs.items():
                    st.progress(prob / 100, text=f"{cls} : {prob:.1f}%")
            else:
                st.session_state.pop('kidney_rejected', None)
                st.session_state['kidney_disease_fr'] = fr
                st.session_state['kidney_confidence'] = conf
                st.session_state['kidney_history'] = []

                colors = {'Normal': '🟢', 'Kyste': '🟡', 'Calcul rénal': '🟠', 'Tumeur': '🔴'}
                st.success(f"{colors.get(fr, '⚪')} **Résultat : {fr}**")
                st.metric("Confiance", f"{conf:.1%}")

                st.markdown("**Probabilités :**")
                for cls, prob in probs.items():
                    st.progress(prob / 100, text=f"{cls} : {prob:.1f}%")


if 'kidney_disease_fr' in st.session_state and 'kidney_rejected' not in st.session_state:
    fr = st.session_state['kidney_disease_fr']
    conf = st.session_state['kidney_confidence']

    st.markdown("---")
    st.subheader("💬 Chatbot Médical")
    st.info(f"🔬 Maladie : **{fr}** | Confiance : **{conf:.1%}**")

    if 'kidney_history' not in st.session_state:
        st.session_state['kidney_history'] = []

    for msg in st.session_state['kidney_history']:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    user_input = st.chat_input("Posez une question sur votre diagnostic...")

    if user_input:
        st.session_state['kidney_history'].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Réflexion..."):
            response = chat_kidney(fr, conf, st.session_state['kidney_history'])

        st.session_state['kidney_history'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state['kidney_response'] = response

    if 'kidney_response' in st.session_state:
        response = st.session_state['kidney_response']

        st.markdown("---")
        st.subheader("🌍 Traduction & Résumé & Audio")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Générer résumé + traduction"):
                with st.spinner("Génération..."):
                    st.session_state['kidney_summary'] = generate_summary(response)
                    st.session_state['kidney_german'] = translate_to_german(st.session_state['kidney_summary'])

        if 'kidney_summary' in st.session_state:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🇫🇷 Résumé")
                st.write(st.session_state['kidney_summary'])
                try:
                    tts = gTTS(st.session_state['kidney_summary'], lang='fr')
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    tts.save(tmp.name)
                    st.audio(tmp.name, format='audio/mp3')
                except:
                    st.warning("Audio indisponible")

            with col2:
                st.markdown("#### 🇩🇪 Zusammenfassung")
                st.write(st.session_state['kidney_german'])
                try:
                    tts = gTTS(st.session_state['kidney_german'], lang='de')
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    tts.save(tmp.name)
                    st.audio(tmp.name, format='audio/mp3')
                except:
                    st.warning("Audio indisponible")
