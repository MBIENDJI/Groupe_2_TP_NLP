# app.py - Version corrigée pour Streamlit Cloud
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import sys

# Configuration
st.set_page_config(
    page_title="Classification des Maladies Rénales",
    page_icon="🏥",
    layout="wide"
)

CLASSES = ['Cyst', 'Normal', 'Stone', 'Tumor']
CLASSES_FR = {0: 'Kyste', 1: 'Normal', 2: 'Calcul rénal', 3: 'Tumeur'}
CONFIDENCE_THRESHOLD = 0.7

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modèle
class KidneyClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(KidneyClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    """Charge le modèle avec recherche automatique"""
    try:
        # Afficher le répertoire courant pour debug
        st.write(f"📁 Répertoire courant: {os.getcwd()}")
        st.write(f"📁 Fichiers présents: {os.listdir('.')}")
        
        # Liste des chemins possibles
        model_paths = [
            "best_model.pth",
            "./best_model.pth",
            os.path.join(os.getcwd(), "best_model.pth"),
            "/mount/src/groupe_2_tp_nlp/best_model.pth",  # Chemin Streamlit Cloud
        ]
        
        model_path = None
        for path in model_paths:
            st.write(f"🔍 Test: {path} - Existe: {os.path.exists(path)}")
            if os.path.exists(path):
                model_path = path
                st.success(f"✅ Modèle trouvé: {path}")
                break
        
        if model_path is None:
            st.error("❌ best_model.pth non trouvé!")
            return None
        
        # Créer le modèle
        model = KidneyClassifier(num_classes=4)
        
        # Charger les poids
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # Nettoyer les clés
        if list(state_dict.keys())[0].startswith('model.'):
            cleaned_dict = {}
            for key, value in state_dict.items():
                new_key = key[6:] if key.startswith('model.') else key
                cleaned_dict[new_key] = value
            state_dict = cleaned_dict
        
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0).to(DEVICE)

def main():
    st.title("🏥 Classification des Maladies Rénales")
    st.markdown("### 🔬 Diagnostic par Intelligence Artificielle (CNN EfficientNet-B0)")
    st.markdown("---")
    
    with st.sidebar:
        st.header("ℹ️ Informations")
        st.write(f"**Modèle:** EfficientNet-B0")
        st.write(f"**Device:** {DEVICE}")
        st.write(f"**Seuil:** {CONFIDENCE_THRESHOLD:.0%}")
    
    # Chargement du modèle
    with st.spinner("🔄 Chargement du modèle..."):
        model = load_model()
    
    if model is None:
        st.stop()
    
    st.success("✅ Modèle chargé avec succès!")
    
    # Interface
    uploaded_file = st.file_uploader("Choisissez une image CT du rein", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        if st.button("Analyser", type="primary"):
            with st.spinner("Analyse..."):
                tensor = preprocess_image(image)
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                
                with col2:
                    if confidence >= CONFIDENCE_THRESHOLD:
                        st.success(f"### ✅ {CLASSES_FR[predicted.item()]}")
                        st.metric("Confiance", f"{confidence.item():.2%}")
                    else:
                        st.warning("### ⚠️ Image non reconnue")
                        st.metric("Confiance max", f"{confidence.item():.2%}")

if __name__ == "__main__":
    main()
