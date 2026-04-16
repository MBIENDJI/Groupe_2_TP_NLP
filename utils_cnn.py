# utils_cnn.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from config import KIDNEY_CLASSES


def load_cnn_model(path="best_model.pth"):
    """Charge le modèle CNN EfficientNet-B0."""
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 4)

    state = torch.load(path, map_location='cpu')

    # Support KidneyClassifier custom
    if any('features' in k for k in state.keys()):
        model.load_state_dict(state)
    else:
        model.load_state_dict(state)

    model.eval()
    return model


def predict_kidney(model, image):
    """
    Prédit la maladie rénale.
    Retourne (classe_en, classe_fr, confidence, probs_dict)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    if isinstance(image, Image.Image):
        img = image.convert('RGB')
    else:
        img = Image.open(image).convert('RGB')

    tensor = transform(img).unsqueeze(0)

    CLASSES_FR = {
        'Cyst'  : 'Kyste',
        'Normal': 'Normal',
        'Stone' : 'Calcul rénal',
        'Tumor' : 'Tumeur'
    }

    with torch.no_grad():
        outputs    = model(tensor)
        probs      = torch.softmax(outputs, dim=1)[0]
        pred_idx   = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    disease_en = KIDNEY_CLASSES[pred_idx]
    disease_fr = CLASSES_FR[disease_en]

    probs_dict = {
        KIDNEY_CLASSES[i]: round(probs[i].item() * 100, 1)
        for i in range(len(KIDNEY_CLASSES))
    }

    return disease_en, disease_fr, confidence, probs_dict
