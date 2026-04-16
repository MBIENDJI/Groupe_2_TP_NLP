# chatbot/monitor.py




import os
chatbot/monitor.py
from datetime import datetime

class LangSmithMonitor:
    """Moniteur local - pas besoin d'API externe"""
    
    def __init__(self):
        self.traces = []
        print("✅ Monitoring actif (mode local)")
    
    def log(self, input_text, output_text, model_name):
        trace = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "input": str(input_text)[:200],
            "output": str(output_text)[:200],
        }
        self.traces.append(trace)
        print(f"[MONITOR] {model_name}: {input_text[:50]}...")
        return trace
    
    def get_stats(self):
        return {
            "total": len(self.traces),
            "models_used": list(set(t["model"] for t in self.traces))
        }

monitor = LangSmithMonitor()
