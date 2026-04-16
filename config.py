# config.py
import os

# ============================================================
# CLAIR NET - SANS CONFUSION
# ============================================================

# 1. GROQ (API principale pour le chat)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# 2. LANG SMITH (monitoring)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "kidney-stock-ai"

# 3. CLASSES POUR LE CNN
KIDNEY_CLASSES = ['Cyst', 'Normal', 'Stone', 'Tumor']
KIDNEY_CLASSES_FR = {
    'Cyst': 'Kyste',
    'Normal': 'Normal',
    'Stone': 'Calcul rénal',
    'Tumor': 'Tumeur'
}

# 4. ENTREPRISES POUR LA BOURSE
COMPANIES = {
    'NVIDIA': 'NVDA',
    'ORACLE': 'ORCL',
    'IBM': 'IBM',
    'CISCO': 'CSCO'
}

# 5. PARAMÈTRES
WINDOW_SIZE = 60
LLM_MODEL = "mixtral-8x7b-32768"  # Modèle Groq ultra-rapide
