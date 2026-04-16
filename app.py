# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np

# Configuration - DOIT être la première commande
st.set_page_config(
    page_title="Classification des Maladies Rénales",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
CLASSES = ['Cyst', 'Normal', 'Stone', 'Tumor']
CLASSES_FR = {
    0: 'Kyste',
    1: 'Normal',
    2: 'Calcul rénal',
    3: 'Tumeur'
}
CONFIDENCE_THRESHOLD = 0.7

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modèle CNN
class KidneyClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(KidneyClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    """Charge le modèle avec mise en cache"""
    try:
        model = KidneyClassifier(num_classes=4)
        
        # Chemins possibles sur Streamlit Cloud
        model_paths = [
            "best_model.pth",
            "./best_model.pth",
            os.path.join(os.path.dirname(__file__), "best_model.pth")
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            st.error("❌ Modèle 'best_model.pth' non trouvé")
            return None
        
        # Charger les poids
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # Nettoyer les clés si elles ont un préfixe 'model.'
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
        st.error(f"❌ Erreur de chargement: {str(e)}")
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
    # Titre
    st.title("🏥 Classification des Maladies Rénales")
    st.markdown("### 🔬 Diagnostic par Intelligence Artificielle (CNN EfficientNet-B0)")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Informations")
        st.markdown(f"""
        - **Modèle:** EfficientNet-B0
        - **Type:** CNN (Convolutional Neural Network)
        - **Accuracy:** 100%
        - **Device:** {DEVICE}
        - **Seuil confiance:** {CONFIDENCE_THRESHOLD:.0%}
        """)
        
        st.markdown("---")
        st.header("📋 Classes diagnostiquées")
        for i, cls in enumerate(CLASSES):
            st.markdown(f"- **{cls}** : {CLASSES_FR[i]}")
        
        st.markdown("---")
        st.header("📖 Mode d'emploi")
        st.markdown("""
        1. 📤 Téléchargez une image CT du rein
        2. 🔬 Cliquez sur 'Analyser'
        3. 📊 Obtenez le diagnostic
        4. 📝 Suivez les recommandations
        """)
        
        st.markdown("---")
        st.caption("⚠️ **Avertissement:** Outil d'aide à la décision. Consultez toujours un médecin.")
    
    # Chargement du modèle
    with st.spinner("🔄 Chargement du modèle en cours..."):
        model = load_model()
    
    if model is None:
        st.error("Impossible de charger le modèle. Vérifiez que best_model.pth est présent.")
        st.stop()
    
    st.success("✅ Modèle chargé avec succès!")
    
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
            
            # Métadonnées
            with st.expander("ℹ️ Métadonnées"):
                st.write(f"**Nom:** {uploaded_file.name}")
                st.write(f"**Taille:** {uploaded_file.size / 1024:.2f} KB")
                st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
        
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
                            # Diagnostic réussi
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
                            # Image non reconnue
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
                    st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
    
    else:
        # Message d'accueil
        st.info("👈 Téléchargez une image CT du rein pour commencer l'analyse")
        
        with st.expander("ℹ️ En savoir plus sur l'application"):
            st.markdown("""
            ### 🧠 Comment ça fonctionne ?
            
            Cette application utilise un **réseau de neurones convolutif (CNN)** 
            de type **EfficientNet-B0** pré-entraîné sur ImageNet et fine-tuné sur 
            plus de 12,000 images CT des reins.
            
            ### 📊 Performance du modèle
            
            | Métrique | Score |
            |----------|-------|
            | Accuracy | 100% |
            | Precision | 100% |
            | Recall | 100% |
            | F1-Score | 100% |
            
            ### ⚠️ Limitations
            
            - Seuil de confiance fixé à 70%
            - Nécessite des images CT de bonne qualité
            - Ne remplace pas un diagnostic médical professionnel
            """)

if __name__ == "__main__":
    main()
