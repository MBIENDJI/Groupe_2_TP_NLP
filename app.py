# app.py (MODIFIÉ avec chatbot)
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import sys

# Ajouter le chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.kidney_chatbot import kidney_chatbot

# Configuration
st.set_page_config(
    page_title="Prédiction des Maladies des Reins",
    page_icon="🏥",
    layout="wide"
)

CLASSES = ['Cyst', 'Normal', 'Stone', 'Tumor']
CLASSES_FR = {0: 'Kyste', 1: 'Normal', 2: 'Calcul rénal', 3: 'Tumeur'}
CONFIDENCE_THRESHOLD = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KidneyClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(KidneyClassifier, self).__init__()
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pth"):
        st.error("❌ best_model.pth non trouvé!")
        return None
    
    try:
        model = KidneyClassifier(num_classes=4)
        state_dict = torch.load("best_model.pth", map_location=DEVICE)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)[:200]}")
        return None

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0).to(DEVICE)

def main():
    st.title("🏥 Prédiction des Maladies des Reins")
    st.markdown("### 🔬 Diagnostic par CNN EfficientNet-B0")
    st.markdown("---")
    
    with st.sidebar:
        st.header("ℹ️ Informations")
        st.write(f"**Modèle:** EfficientNet-B0")
        st.write(f"**Device:** {DEVICE}")
        st.write(f"**Seuil de confiance:** {CONFIDENCE_THRESHOLD:.0%}")
        st.markdown("---")
        st.header("📋 Classes")
        st.write("- **Cyst** : Kyste")
        st.write("- **Normal** : Normal")
        st.write("- **Stone** : Calcul rénal")
        st.write("- **Tumor** : Tumeur")
    
    model = load_model()
    if model is None:
        st.stop()
    
    uploaded_file = st.file_uploader(
        "📤 Choisissez une image CT du rein",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléchargée", use_container_width=True)
        
        if st.button("🔬 Analyser l'image", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyse en cours..."):
                image_tensor = preprocess_image(image)
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                
                diagnosis = CLASSES_FR[predicted.item()]
                
                with col2:
                    st.subheader("📊 Résultat")
                    st.progress(confidence.item())
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        st.success(f"### ✅ Diagnostic: {diagnosis}")
                        st.metric("Confiance", f"{confidence:.2%}")
                        
                        if predicted.item() == 1:
                            recommendation = "Résultat normal. Continuez les contrôles réguliers."
                            st.info(recommendation)
                        elif predicted.item() == 0:
                            recommendation = "Consultez un néphrologue pour évaluation."
                            st.warning(recommendation)
                        elif predicted.item() == 2:
                            recommendation = "Consultez un urologue. Une intervention peut être nécessaire."
                            st.error(recommendation)
                        else:
                            recommendation = "Consultation urgente avec un oncologue recommandée."
                            st.error(recommendation)
                    else:
                        recommendation = "Image non reconnue. Veuillez consulter un radiologue."
                        st.warning(f"### ⚠️ {recommendation}")
                        st.metric("Confiance max", f"{confidence:.2%}")
                    
                    with st.expander("📈 Détail des probabilités"):
                        for i, cls in enumerate(CLASSES):
                            prob = probs[0][i].item()
                            st.write(f"**{cls}**: {prob:.2%}")
                            st.progress(prob)
                
                # ============================================================
                # CHATBOT INTELLIGENT
                # ============================================================
                if confidence >= CONFIDENCE_THRESHOLD:
                    chatbot_result = kidney_chatbot.process_result(
                        diagnosis=diagnosis,
                        confidence=confidence,
                        recommendation=recommendation
                    )
                    kidney_chatbot.display_chatbot_ui(chatbot_result)
    
    else:
        st.info("👈 Téléchargez une image CT du rein pour commencer")

if __name__ == "__main__":
    main()
