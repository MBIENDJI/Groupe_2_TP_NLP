# app.py - Version CORRIGÉE avec la bonne structure
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Configuration
st.set_page_config(
    page_title="Classification des Maladies Rénales",
    page_icon="🏥",
    layout="wide"
)

CLASSES = ['Cyst', 'Normal', 'Stone', 'Tumor']
CLASSES_FR = {0: 'Kyste', 1: 'Normal', 2: 'Calcul rénal', 3: 'Tumeur'}
CONFIDENCE_THRESHOLD = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modèle CORRIGÉ - Structure identique au modèle sauvegardé
class KidneyClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(KidneyClassifier, self).__init__()
        # Charger EfficientNet-B0 pré-entraîné
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Prendre les features (sans le classifier)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        # Créer le classifier
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
    """Charge le modèle avec la bonne structure"""
    try:
        # Afficher le répertoire courant
        st.write(f"📁 Répertoire courant: {os.getcwd()}")
        
        # Vérifier si le fichier existe
        if not os.path.exists("best_model.pth"):
            st.error("❌ best_model.pth non trouvé!")
            return None
        
        st.success("✅ best_model.pth trouvé!")
        
        # Créer le modèle
        model = KidneyClassifier(num_classes=4)
        
        # Charger les poids directement (sans modification)
        state_dict = torch.load("best_model.pth", map_location=DEVICE)
        
        # Afficher les premières clés pour debug
        st.write(f"🔑 Clés du modèle: {list(state_dict.keys())[:3]}")
        
        # Charger les poids
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        
        st.success("✅ Modèle chargé avec succès!")
        return model
        
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)[:200]}")
        return None

def preprocess_image(image):
    """Prétraite l'image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        st.write(f"**Seuil de confiance:** {CONFIDENCE_THRESHOLD:.0%}")
        
        st.markdown("---")
        st.header("📋 Classes")
        st.write("- **Cyst** : Kyste")
        st.write("- **Normal** : Normal")
        st.write("- **Stone** : Calcul rénal")
        st.write("- **Tumor** : Tumeur")
    
    # Chargement du modèle
    with st.spinner("🔄 Chargement du modèle en cours..."):
        model = load_model()
    
    if model is None:
        st.stop()
    
    # Interface principale
    uploaded_file = st.file_uploader(
        "📤 Choisissez une image CT du rein",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Formats supportés: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléchargée", use_container_width=True)
        
        if st.button("🔬 Analyser l'image", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyse en cours..."):
                try:
                    # Prétraitement
                    image_tensor = preprocess_image(image)
                    
                    # Prédiction
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                    
                    with col2:
                        st.subheader("📊 Résultat de l'analyse")
                        
                        # Barre de confiance
                        st.write("**Niveau de confiance:**")
                        st.progress(confidence.item())
                        st.caption(f"{confidence.item():.2%}")
                        
                        if confidence >= CONFIDENCE_THRESHOLD:
                            st.success(f"### ✅ Diagnostic: {CLASSES_FR[predicted.item()]}")
                            st.metric("Confiance", f"{confidence.item():.2%}")
                            
                            st.markdown("---")
                            st.write("**📋 Recommandations:**")
                            
                            if predicted.item() == 1:  # Normal
                                st.info("✅ Résultat normal. Continuez les contrôles réguliers.")
                            elif predicted.item() == 0:  # Cyst
                                st.warning("⚠️ Kyste détecté. Consultez un néphrologue.")
                            elif predicted.item() == 2:  # Stone
                                st.error("⚠️ Calcul rénal détecté. Consultez un urologue.")
                            else:  # Tumor
                                st.error("🚨 Tumeur détectée. Consultation urgente avec un oncologue.")
                        else:
                            st.warning("### ⚠️ Image non reconnue")
                            st.metric("Confiance maximale", f"{confidence.item():.2%}")
                            st.write(f"**Seuil requis:** {CONFIDENCE_THRESHOLD:.0%}")
                            
                            st.markdown("---")
                            st.write("**🔍 Causes possibles:**")
                            st.write("- ❌ L'image n'est pas une CT du rein")
                            st.write("- 📷 Qualité d'image insuffisante")
                            st.write("- 🎯 Angle ou coupe non standard")
                        
                        # Détails des probabilités
                        with st.expander("📈 Détail des probabilités par classe"):
                            for i, cls in enumerate(CLASSES):
                                prob = probabilities[0][i].item()
                                st.write(f"**{cls}** ({CLASSES_FR[i]}): {prob:.2%}")
                                st.progress(prob)
                
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
    
    else:
        st.info("👈 Téléchargez une image CT du rein pour commencer l'analyse")

if __name__ == "__main__":
    main()
