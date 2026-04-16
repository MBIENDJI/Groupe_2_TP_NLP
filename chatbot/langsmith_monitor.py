# chatbot/langsmith_monitor.py
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class LangSmithMonitor:
    """Monitoring des interactions avec LangSmith"""
    
    def __init__(self):
        self.api_key = os.getenv("LANGSMITH_API_KEY")
        self.project = os.getenv("LANGSMITH_PROJECT", "kidney_stock_chatbot")
        self.traces = []
        
    def log_interaction(self, input_data, output_data, model_name, success=True):
        """Enregistre une interaction"""
        trace = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "input": input_data,
            "output": output_data,
            "success": success,
            "project": self.project
        }
        self.traces.append(trace)
        
        # Si API key présente, envoyer à LangSmith
        if self.api_key:
            self._send_to_langsmith(trace)
        
        return trace
    
    def _send_to_langsmith(self, trace):
        """Envoie la trace à LangSmith (simulation)"""
        # Dans un environnement réel, utiliser requests.post
        print(f"[LangSmith] Trace envoyée: {trace['model']} - {trace['timestamp']}")
    
    def get_stats(self):
        """Retourne les statistiques"""
        return {
            "total_interactions": len(self.traces),
            "success_rate": sum(1 for t in self.traces if t["success"]) / len(self.traces) if self.traces else 0,
            "models_used": list(set(t["model"] for t in self.traces))
        }

# Instance globale
monitor = LangSmithMonitor()
