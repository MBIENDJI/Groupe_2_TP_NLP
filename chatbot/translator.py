# chatbot/translator.py

class GermanTranslator:
    """Traducteur français → allemand"""
    
    # Dictionnaire de traduction médicale
    MEDICAL_TERMS = {
        "kyste": "Zyste",
        "normal": "normal",
        "calcul rénal": "Nierenstein",
        "tumeur": "Tumor",
        "rein": "Niere",
        "maladie": "Krankheit",
        "diagnostic": "Diagnose",
        "confiance": "Vertrauen",
        "recommandation": "Empfehlung",
        "consultez": "Konsultieren Sie",
        "néphrologue": "Nephrologen",
        "urologue": "Urologen",
        "oncologue": "Onkologen",
        "urgence": "Notfall"
    }
    
    COMMON_PHRASES = {
        "Résultat normal": "Normales Ergebnis",
        "Kyste détecté": "Zyste erkannt",
        "Calcul rénal détecté": "Nierenstein erkannt",
        "Tumeur détectée": "Tumor erkannt",
        "Consultation urgente recommandée": "Dringende Konsultation empfohlen"
    }
    
    @classmethod
    def translate(cls, text):
        """Traduit un texte français en allemand"""
        result = text
        
        # Traduire les phrases communes
        for fr, de in cls.COMMON_PHRASES.items():
            result = result.replace(fr, de)
        
        # Traduire les termes médicaux
        for fr, de in cls.MEDICAL_TERMS.items():
            result = result.replace(fr, de)
        
        return result
    
    @classmethod
    def translate_result(cls, diagnosis, confidence, recommendation):
        """Traduit un résultat complet"""
        translation = {
            "diagnosis": cls.translate(diagnosis),
            "confidence": f"Vertrauen: {confidence:.1%}",
            "recommendation": cls.translate(recommendation),
            "original_french": {
                "diagnosis": diagnosis,
                "confidence": f"Confiance: {confidence:.1%}",
                "recommendation": recommendation
            }
        }
        return translation
