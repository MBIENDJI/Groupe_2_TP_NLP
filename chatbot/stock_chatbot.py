# chatbot/stock_chatbot.py
import streamlit as st
from .langsmith_monitor import monitor

class StockChatbot:
    """Chatbot intelligent pour les prédictions boursières"""
    
    def __init__(self):
        self.conversation_history = []
    
    def process_query(self, company, predictions, current_price):
        """Traite la requête utilisateur sur une entreprise"""
        
        # Générer la réponse
        response = self._generate_response(company, predictions, current_price)
        
        # Monitoring LangSmith
        monitor.log_interaction(
            input_data=f"Company: {company}",
            output_data=response[:500],
            model_name="Stock_Chatbot"
        )
        
        return {
            "response": response,
            "company": company,
            "predictions": predictions,
            "current_price": current_price
        }
    
    def _generate_response(self, company, predictions, current_price):
        """Génère une réponse personnalisée"""
        
        response = f"""
        🤖 **Analyse Financière - {company}**
        
        📊 **Prix actuel:** ${current_price:.2f}
        
        📈 **Prédictions des modèles:**
        """
        
        for model, data in predictions.items():
            variation = data['variation']
            if variation > 0:
                trend = "📈 hausse"
            else:
                trend = "📉 baisse"
            response += f"\n   • **{model}:** ${data['final_price']:.2f} ({trend} de {abs(variation):.1f}%)"
        
        # Analyse et recommandation
        response += f"""
        
        💡 **Analyse:**
        """
        
        # Trouver le consensus
        variations = [data['variation'] for data in predictions.values()]
        avg_variation = sum(variations) / len(variations) if variations else 0
        
        if avg_variation > 5:
            response += "\n   📈 **Tendance haussière forte** - Perspective très positive"
            response += "\n   🟢 **Recommandation:** ACHAT FORT"
        elif avg_variation > 2:
            response += "\n   📈 **Tendance haussière modérée** - Perspective favorable"
            response += "\n   🟡 **Recommandation:** ACHAT"
        elif avg_variation > -2:
            response += "\n   📊 **Tendance neutre** - Marché stable"
            response += "\n   ⚪ **Recommandation:** CONSERVER"
        elif avg_variation > -5:
            response += "\n   📉 **Tendance baissière modérée** - Prudence recommandée"
            response += "\n   🟠 **Recommandation:** VENDRE LÉGÈREMENT"
        else:
            response += "\n   📉 **Tendance baissière forte** - Signal négatif"
            response += "\n   🔴 **Recommandation:** VENDRE"
        
        response += f"""
        
        ⚠️ **Avertissement:** Ces prédictions sont basées sur des modèles d'IA. Ne constitue pas un conseil financier.
        """
        
        return response
    
    def display_chatbot_ui(self, result):
        """Affiche l'interface du chatbot dans Streamlit"""
        
        st.markdown("---")
        st.subheader("🤖 Assistant Financier Intelligent")
        
        # Zone de questions supplémentaires
        user_question = st.text_input("💬 Posez une question sur cette prédiction:", 
                                      placeholder="Ex: Quelle est la tendance à long terme?")
        
        if user_question:
            self._handle_followup(user_question, result)
        
        # Afficher la réponse principale
        st.markdown(result["response"])
    
    def _handle_followup(self, question, result):
        """Gère les questions complémentaires"""
        
        # Réponses basiques aux questions fréquentes
        question_lower = question.lower()
        
        if "tendance" in question_lower or "long terme" in question_lower:
            st.info("📈 La tendance à long terme dépend des fondamentaux de l'entreprise. Les modèles IA prévoient une tendance basée sur l'historique 5 ans.")
        elif "risque" in question_lower:
            st.warning("⚠️ Tout investissement comporte des risques. Diversifiez votre portefeuille.")
        elif "confiance" in question_lower:
            st.info(f"📊 Le modèle avec la meilleure performance est basé sur son MAPE (erreur de prédiction).")
        else:
            st.info("💡 Pour une analyse plus approfondie, consultez un conseiller financier.")

# Instance globale
stock_chatbot = StockChatbot()
