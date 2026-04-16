# config.py
import os

# ============================================================
# GROQ - Modèle actif (avril 2026)
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"  # Modèle actif sur Groq

# ============================================================
# LANG SMITH (monitoring)
# ============================================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "kidney-stock-ai"

# ============================================================
# CNN - Classification maladies rénales
# ============================================================
KIDNEY_CLASSES = ['Cyst', 'Normal', 'Stone', 'Tumor']
KIDNEY_CLASSES_FR = {
    'Cyst': 'Kyste',
    'Normal': 'Normal',
    'Stone': 'Calcul rénal',
    'Tumor': 'Tumeur'
}

# ============================================================
# STOCKS - Entreprises
# ============================================================
COMPANIES = {
    'NVIDIA': 'NVDA',
    'ORACLE': 'ORCL',
    'IBM': 'IBM',
    'CISCO': 'CSCO'
}

WINDOW_SIZE = 60
CONFIDENCE_THRESHOLD = 0.70
