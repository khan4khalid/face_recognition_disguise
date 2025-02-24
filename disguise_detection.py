import os
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import pickle

# Paths
model_save_path = "models/vit_finetuned_disguise"
class_labels_path = "models/class_labels.pkl"

# Load the trained model
model = ViTForImageClassification.from_pretrained(model_save_path)
model.eval()

# Load class labels
with open(class_labels_path, "rb") as f:
    class_labels = pickle.load(f)
id_to_class = {v: k for k, v in class_labels.items()}

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def detect_disguise(image_path):
    """
    Detects the disguise type from an input image.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        str: Detected disguise type.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    # Map class ID to class name
    predicted_class_name = id_to_class[predicted_class_id]
    return predicted_class_name

# Example usage
image_path = "0003.jpg"  # Replace with your test image
disguise_type = detect_disguise(image_path)
print(f"Detected Disguise: {disguise_type}")
