# chatbot/summarizer.py

class Summarizer:
    """Générateur de résumé pour les résultats"""
    
    @staticmethod
    def summarize_kidney_result(diagnosis, confidence, recommendation):
        """Résumé du résultat CNN"""
        
        summaries = {
            "Normal": "Le rein est sain. Aucune anomalie détectée. Continuez les contrôles réguliers.",
            "Kyste": "Un kyste rénal a été identifié. Il s'agit généralement d'une lésion bénigne nécessitant une surveillance.",
            "Calcul rénal": "Un calcul rénal est présent. Une prise en charge urologique est recommandée.",
            "Tumeur": "Une tumeur rénale suspecte a été détectée. Une consultation oncologique urgente est nécessaire."
        }
        
        base_summary = summaries.get(diagnosis, f"Diagnostic: {diagnosis}")
        
        full_summary = f"""
        📋 RÉSUMÉ DU DIAGNOSTIC
        ─────────────────────────
        🩺 Diagnostic: {diagnosis}
        📊 Confiance: {confidence:.1%}
        💡 {base_summary}
        🏥 Action: {recommendation}
        """
        
        return {
            "short": base_summary,
            "detailed": full_summary,
            "diagnosis": diagnosis,
            "confidence": confidence
        }
    
    @staticmethod
    def summarize_stock_prediction(company, predictions, current_price):
        """Résumé des prédictions boursières"""
        
        best_pred = max(predictions.items(), key=lambda x: x[1]['variation']) if predictions else None
        
        summary = f"""
        📊 RÉSUMÉ PRÉDICTION {company.upper()}
        ─────────────────────────────────
        💰 Prix actuel: ${current_price:.2f}
        """
        
        for model, data in predictions.items():
            summary += f"\n   🔮 {model}: ${data['final_price']:.2f} ({data['variation']:+.1f}%)"
        
        if best_pred:
            summary += f"\n\n🏆 Meilleure prédiction: {best_pred[0]} (+{best_pred[1]['variation']:.1f}%)"
        
        return summary
