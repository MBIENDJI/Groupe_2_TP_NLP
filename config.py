# config.py
import os

KIDNEY_CLASSES    = ['Cyst', 'Normal', 'Stone', 'Tumor']
KIDNEY_CLASSES_FR = {
    'Cyst'  : 'Kyste',
    'Normal': 'Normal',
    'Stone' : 'Calcul rénal',
    'Tumor' : 'Tumeur'
}

COMPANIES = {
    'NVIDIA': 'NVDA',
    'ORACLE': 'ORCL',
    'IBM'   : 'IBM',
    'CISCO' : 'CSCO'
}

WINDOW_SIZE = 60

# Cherche sam_token EN PREMIER (ton secret Streamlit Cloud)
# puis HF_TOKEN en fallback
HF_TOKEN  = (os.environ.get("sam_token") or
             os.environ.get("HF_TOKEN") or "")

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # ou "HuggingFaceH4/zephyr-7b-beta"
# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.environ.get(
    "LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = "kidney-stock-ai"
