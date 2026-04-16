# chatbot/llm_groq.pyimport os

import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)

def call_llm(messages, max_tokens=500):
    if not GROQ_API_KEY:
        return "⚠️ Clé Groq manquante"
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Erreur: {str(e)[:100]}"

def chat_kidney(disease_fr, confidence, conversation_history):
    system_msg = {
        "role": "system",
        "content": f"Assistant médical. Diagnostic: {disease_fr} (confiance {confidence:.1%}). Réponds en français simplement."
    }
    last_msg = conversation_history[-1]["content"] if conversation_history else "Explique"
    messages = [system_msg, {"role": "user", "content": last_msg}]
    return call_llm(messages)

def chat_stock(company, months, lstm_pred, prophet_pred, neural_pred, conversation_history):
    system_msg = {
        "role": "system",
        "content": f"Analyste financier. {company} sur {months} mois. LSTM: ${lstm_pred:.2f}, Prophet: ${prophet_pred:.2f}, NeuralProphet: ${neural_pred:.2f}. Réponds en français."
    }
    last_msg = conversation_history[-1]["content"] if conversation_history else "Analyse"
    messages = [system_msg, {"role": "user", "content": last_msg}]
    return call_llm(messages)

def translate_to_german(text):
    messages = [
        {"role": "system", "content": "Traduis en allemand. Donne uniquement la traduction."},
        {"role": "user", "content": text}
    ]
    return call_llm(messages, max_tokens=500)

def generate_summary(text):
    messages = [
        {"role": "system", "content": "Résumé court en français (3-4 phrases)."},
        {"role": "user", "content": text}
    ]
    return call_llm(messages, max_tokens=300)
