# chatbot/llm.py
import requests
import os
from config import HF_TOKEN, LLM_MODEL

# Importer monitor APRÈS sa définition (pas d'import circulaire)
# from chatbot.monitor import monitor  ← À SUPPRIMER


def call_llm(messages, max_tokens=500):
    """
    Appel HuggingFace Inference API - Version CORRECTE
    """
    if not HF_TOKEN:
        return "⚠️ Token Hugging Face manquant. Configurez HF_TOKEN dans les secrets."

    # URL CORRECTE (sans /v1/chat/completions)
    url = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Construire le prompt à partir des messages
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"Instruction: {msg['content']}\n\n"
        elif msg["role"] == "user":
            prompt += f"Question: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Réponse: {msg['content']}\n"
    
    prompt += "Réponse: "
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "do_sample": True
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "Désolé, je n'ai pas pu générer de réponse.")
            elif isinstance(result, dict):
                return result.get("generated_text", "Désolé, je n'ai pas pu générer de réponse.")
            else:
                return str(result)
        else:
            return f"⚠️ Service temporairement indisponible (code {response.status_code})"
            
    except Exception as e:
        return f"⚠️ Erreur de connexion : Le service est momentanément indisponible."


def chat_kidney(disease_fr, confidence, conversation_history):
    """Chatbot médical pour résultat CNN."""
    
    system_msg = (
        f"Tu es un assistant médical spécialisé en néphrologie. "
        f"L'analyse par intelligence artificielle a détecté : {disease_fr} "
        f"avec une confiance de {confidence:.1%}. "
        f"Réponds de manière simple, claire et empathique en français. "
        f"Explique ce qu'est cette maladie en termes simples. "
        f"Recommande toujours de consulter un médecin. "
        f"N'utilise jamais de jargon médical compliqué."
    )
    
    # Récupérer le dernier message de l'utilisateur
    last_user_msg = "Explique-moi ce diagnostic"
    for msg in reversed(conversation_history):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": last_user_msg}
    ]

    response = call_llm(messages)
    
    return response


def chat_stock(company, months, lstm_pred, prophet_pred, neural_pred, conversation_history):
    """Chatbot financier pour prédictions boursières."""
    
    system_msg = (
        f"Tu es un conseiller financier pédagogique. "
        f"Pour {company}, les modèles prévoient sur {months} mois : "
        f"LSTM à ${lstm_pred:.2f}, Prophet à ${prophet_pred:.2f}, "
        f"NeuralProphet à ${neural_pred:.2f}. "
        f"Explique simplement ce que cela signifie. "
        f"Rappelle toujours que ce n'est pas un conseil financier. "
        f"Réponds en français."
    )
    
    last_user_msg = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": last_user_msg}
    ]

    response = call_llm(messages)
    
    return response


def translate_to_german(text):
    """Traduit un texte en allemand."""
    messages = [
        {"role": "system", "content": "Tu es un traducteur. Traduis le texte suivant en allemand. Donne uniquement la traduction, rien d'autre."},
        {"role": "user", "content": f"À traduire en allemand : {text}"}
    ]
    result = call_llm(messages, max_tokens=500)
    return result


def generate_summary(text):
    """Génère un résumé court en français."""
    messages = [
        {"role": "system", "content": "Tu résumes des textes médicaux. Fais un résumé court (3-4 phrases) en français."},
        {"role": "user", "content": f"Résume ceci : {text}"}
    ]
    result = call_llm(messages, max_tokens=300)
    return result
