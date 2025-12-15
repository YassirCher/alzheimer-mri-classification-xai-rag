import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
from flask import current_app

class AlzheimerModel:
    def __init__(self):
        self.device = torch.device(current_app.config['DEVICE'] if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.class_names = current_app.config['CLASS_NAMES']
        self.load_model()
        self.setup_transforms()
    
    def load_model(self):
        """Load trained model"""
        try:
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, 4)
            
            model.load_state_dict(torch.load(
                current_app.config['MODEL_PATH'],
                map_location=self.device
            ))
            model.to(self.device)
            model.eval()
            self.model = model
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def setup_transforms(self):
        """Setup image transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_input):
        """
        Predict on image input
        image_input: PIL Image or file path
        Returns: class_idx, class_name, probabilities
        """
        try:
            # Load image
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            else:
                img = image_input.convert('RGB')
            
            # Transform and batch
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
                all_probs = probs[0].cpu().numpy()
            
            return {
                'class_idx': pred_class,
                'class_name': self.class_names[pred_class],
                'confidence': float(confidence),
                'probabilities': {self.class_names[i]: float(all_probs[i]) for i in range(4)},
                'image': img
            }
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            raise

# Global model instance
_model_instance = None

def get_model():
    """Get or create model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = AlzheimerModel()
    return _model_instance
