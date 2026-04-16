# chatbot/kidney_chatbot.py
import streamlit as st
from .translator import GermanTranslator
from .summarizer import Summarizer
from .text_to_speech import tts
from .langsmith_monitor import monitor

class KidneyChatbot:
    """Chatbot intelligent pour les résultats de classification CNN"""
    
    def __init__(self):
        self.conversation_history = []
    
    def process_result(self, diagnosis, confidence, recommendation):
        """Traite le résultat du CNN et génère une réponse"""
        
        # Générer la réponse du chatbot
        response = self._generate_response(diagnosis, confidence, recommendation)
        
        # Traduction en allemand
        german_response = GermanTranslator.translate(response)
        
        # Résumé
        summary = Summarizer.summarize_kidney_result(diagnosis, confidence, recommendation)
        
        # Monitoring LangSmith
        monitor.log_interaction(
            input_data=f"Diagnosis: {diagnosis}, Confidence: {confidence}",
            output_data=response,
            model_name="Kidney_CNN_Chatbot"
        )
        
        # Audio
        audio_file = tts.to_audio(response, f"kidney_result_{diagnosis}.mp3")
        
        return {
            "response": response,
            "german": german_response,
            "summary": summary,
            "audio_path": audio_file,
            "diagnosis": diagnosis,
            "confidence": confidence
        }
    
    def _generate_response(self, diagnosis, confidence, recommendation):
        """Génère une réponse personnalisée"""
        
        responses = {
            "Normal": f"""
            🤖 **Analyse du Chatbot Médical**
            
            📊 **Résultat:** Le modèle CNN a analysé l'image avec une confiance de {confidence:.1%}.
            
            ✅ **Diagnostic:** Rein normal détecté.
            
            💬 **Interprétation:** L'image CT montre un rein sain sans anomalie visible. Les structures rénales sont normales.
            
            📋 **Recommandation:** {recommendation}
            
            🔍 **Conseil:** Continuez les bilans de santé réguliers. Pas d'inquiétude particulière.
            """,
            
            "Kyste": f"""
            🤖 **Analyse du Chatbot Médical**
            
            📊 **Résultat:** Le modèle CNN a analysé l'image avec une confiance de {confidence:.1%}.
            
            🩺 **Diagnostic:** Kyste rénal détecté.
            
            💬 **Interprétation:** Une cavité remplie de liquide (kyste) a été identifiée sur l'image CT. La plupart des kystes rénaux sont bénins.
            
            📋 **Recommandation:** {recommendation}
            
            🔍 **Conseil:** Une échographie de contrôle peut être proposée pour confirmer la nature bénigne.
            """,
            
            "Calcul rénal": f"""
            🤖 **Analyse du Chatbot Médical**
            
            📊 **Résultat:** Le modèle CNN a analysé l'image avec une confiance de {confidence:.1%}.
            
            🪨 **Diagnostic:** Calcul rénal détecté.
            
            💬 **Interprétation:** Une formation solide (calcul) a été identifiée dans le système urinaire. Cela peut causer des douleurs.
            
            📋 **Recommandation:** {recommendation}
            
            🔍 **Conseil:** Une prise en charge rapide est recommandée pour éviter les complications.
            """,
            
            "Tumeur": f"""
            🤖 **Analyse du Chatbot Médical - ⚠️ ALERTE ⚠️**
            
            📊 **Résultat:** Le modèle CNN a analysé l'image avec une confiance de {confidence:.1%}.
            
            🚨 **Diagnostic:** Tumeur rénale suspecte détectée.
            
            💬 **Interprétation:** Une masse anormale a été identifiée. Une caractérisation radiologique et histologique est nécessaire.
            
            📋 **Recommandation:** {recommendation}
            
            🔍 **Conseil:** Ne tardez pas à consulter un spécialiste pour prise en charge rapide.
            """
        }
        
        return responses.get(diagnosis, f"Diagnostic: {diagnosis}")
    
    def display_chatbot_ui(self, result):
        """Affiche l'interface du chatbot dans Streamlit"""
        
        st.markdown("---")
        st.subheader("🤖 Assistant Médical Intelligent")
        
        # Tabs pour les différentes fonctionnalités
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Réponse", "🇩🇪 Allemand", "📝 Résumé", "🎵 Audio"])
        
        with tab1:
            st.markdown(result["response"])
        
        with tab2:
            st.markdown(f"### 🇩🇪 Traduction allemande")
            st.markdown(result["german"])
        
        with tab3:
            st.markdown(result["summary"]["detailed"])
        
        with tab4:
            if result["audio_path"] and os.path.exists(result["audio_path"]):
                st.audio(result["audio_path"])
            else:
                st.info("🎙️ Audio généré automatiquement")

# Instance globale
kidney_chatbot = KidneyChatbot()
