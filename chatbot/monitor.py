# chatbot/monitor.py
import os
from datetime import datetime

class LangSmithMonitor:
    """Monitoring LangSmith des interactions."""

    def __init__(self):
        self.api_key = os.environ.get("LANGCHAIN_API_KEY", "")
        self.project = os.environ.get(
            "LANGCHAIN_PROJECT", "kidney-stock-ai")
        self.traces  = []

    def log(self, input_text, output_text, model_name):
        """Enregistre une interaction."""
        trace = {
            "timestamp" : datetime.now().isoformat(),
            "model"     : model_name,
            "input"     : str(input_text)[:500],
            "output"    : str(output_text)[:500],
            "project"   : self.project
        }
        self.traces.append(trace)

        if self.api_key:
            try:
                from langsmith import Client
                client = Client()
                client.create_run(
                    name     = model_name,
                    run_type = "llm",
                    inputs   = {"input" : trace["input"]},
                    outputs  = {"output": trace["output"]}
                )
            except Exception as e:
                print(f"LangSmith log error: {e}")

        return trace

    def get_stats(self):
        """Retourne les statistiques."""
        total = len(self.traces)
        return {
            "total"      : total,
            "models_used": list(set(
                t["model"] for t in self.traces))
        }


# Instance globale
monitor = LangSmithMonitor()
