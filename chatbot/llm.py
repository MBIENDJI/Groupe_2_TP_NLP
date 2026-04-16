# chatbot/llm.py
import requests
import os
from config import HF_TOKEN, LLM_MODEL
from chatbot.monitor import monitor


def call_llm(messages, max_tokens=800):
    """
    Appel HuggingFace Inference API — URL corrigée.
    L'endpoint /v1/chat/completions n'est PAS dans /models/
    il est sous api-inference.huggingface.co/v1/
    """
    if not HF_TOKEN:
        return "⚠️ HF_TOKEN manquant. Configurez le secret."

    # URL CORRECTE pour l'API conversationnelle HF
    url = "https://api-inference.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type" : "application/json"
    }
    payload = {
        "model"      : LLM_MODEL,
        "messages"   : messages,
        "max_tokens" : max_tokens,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            url, headers=headers,
            json=payload, timeout=60
        )
        if response.status_code == 200:
            return (response.json()
                    ["choices"][0]["message"]["content"])
        else:
            return (f"Erreur API {response.status_code}: "
                    f"{response.text[:300]}")
    except Exception as e:
        return f"Erreur de connexion : {str(e)}"


def chat_kidney(disease_fr, confidence,
                conversation_history):
    system = {
        "role"   : "system",
        "content": (
            f"Tu es un assistant médical spécialisé "
            f"en néphrologie. Le CNN a détecté : "
            f"{disease_fr} avec une confiance de "
            f"{confidence:.1%}. "
            f"Réponds toujours en français. "
            f"Sois empathique, clair et recommande "
            f"toujours de consulter un médecin. "
            f"Ne pose jamais de diagnostic définitif."
        )
    }
    messages = [system] + conversation_history
    response = call_llm(messages)
    monitor.log(
        input_text  = conversation_history[-1]["content"]
                      if conversation_history else "",
        output_text = response,
        model_name  = "kidney_chatbot"
    )
    return response


def chat_stock(company, months, lstm_pred,
               prophet_pred, neural_pred,
               conversation_history):
    system = {
        "role"   : "system",
        "content": (
            f"Tu es un analyste financier expert. "
            f"Analyse les prédictions pour {company}. "
            f"Période : {months} mois. "
            f"LSTM : ${lstm_pred:.2f} | "
            f"Prophet : ${prophet_pred:.2f} | "
            f"NeuralProphet : ${neural_pred:.2f}. "
            f"Réponds en français. "
            f"Avertis toujours que ce n'est pas "
            f"un conseil financier."
        )
    }
    messages = [system] + conversation_history
    response = call_llm(messages)
    monitor.log(
        input_text  = conversation_history[-1]["content"]
                      if conversation_history else "",
        output_text = response,
        model_name  = "stock_chatbot"
    )
    return response


def translate_to_german(text):
    messages = [{
        "role"   : "user",
        "content": (
            f"Traduis ce texte en allemand. "
            f"Retourne uniquement la traduction "
            f"sans commentaire :\n\n{text}"
        )
    }]
    result = call_llm(messages, max_tokens=500)
    monitor.log(text, result, "translation_de")
    return result


def generate_summary(text):
    messages = [{
        "role"   : "user",
        "content": (
            f"Fais un résumé en 3-4 phrases maximum "
            f"en français de ce texte :\n\n{text}"
        )
    }]
    result = call_llm(messages, max_tokens=300)
    monitor.log(text, result, "summarizer")
    return result
