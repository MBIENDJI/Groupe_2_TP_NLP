# config.py
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Modèle actif et disponible sur Groq
LLM_MODEL = "llama3-70b-8192"  # ou "llama3-8b-8192" ou "gemma2-9b-it"

# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "kidney-stock-ai"

# Classes
KIDNEY_CLASSES = ['Cyst', 'Normal', 'Stone', 'Tumor']
KIDNEY_CLASSES_FR = {
    'Cyst': 'Kyste',
    'Normal': 'Normal',
    'Stone': 'Calcul rénal',
    'Tumor': 'Tumeur'
}

COMPANIES = {
    'NVIDIA': 'NVDA',
    'ORACLE': 'ORCL',
    'IBM': 'IBM',
    'CISCO': 'CSCO'
}

WINDOW_SIZE = 60
