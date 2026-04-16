# chatbot/text_to_speech.py
import os
import platform

class TextToSpeech:
    """Convertit le texte en audio"""
    
    def __init__(self, output_dir="audio_outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def to_audio(self, text, filename="output.mp3"):
        """Convertit le texte en audio (simulation)"""
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Selon le système d'exploitation
        system = platform.system()
        
        if system == "Windows":
            # Utiliser pyttsx3 ou gTTS
            try:
                from gtts import gTTS
                tts = gTTS(text=text, lang='fr')
                tts.save(filepath)
                return filepath
            except ImportError:
                # Simuler si gTTS non installé
                self._create_dummy_audio(filepath)
                return filepath
        else:
            # Linux/Mac
            self._create_dummy_audio(filepath)
            return filepath
    
    def _create_dummy_audio(self, filepath):
        """Crée un fichier audio factice pour test"""
        with open(filepath, 'w') as f:
            f.write("Audio content placeholder")
    
    def get_audio_html(self, filepath):
        """Génère le HTML pour lire l'audio dans Streamlit"""
        if os.path.exists(filepath):
            return f"""
            <audio controls>
                <source src="{filepath}" type="audio/mpeg">
                Votre navigateur ne supporte pas l'audio.
            </audio>
            """
        return "Audio non disponible"

tts = TextToSpeech()
