import os
from typing import Dict, Any
from PIL import Image
import torch
from torchvision import transforms
from src.model import BrainTumorClassifier
from src.dataset import NORM_MEAN, NORM_STD
from src.gradcam import GradCAM, overlay_gradcam

DEFAULT_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

class BrainTumorPredictor:
    """
    High-level Inference engine for Brain Tumor MRI scans.
    """
    def __init__(self, checkpoint_path: str, device: str = None, classes: list = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = classes if classes is not None else DEFAULT_CLASSES

        self.model = BrainTumorClassifier(num_classes=len(self.classes), pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])

    def predict(self, image_path: str, return_gradcam: bool = True) -> Dict[str, Any]:
        """
        Runs inference on an MRI image and returns diagnosis and probabilities.
        """
        img_pil = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        pred_label = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        all_probs = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}

        result = {
            'predicted_class': pred_label,
            'confidence': confidence,
            'probabilities': all_probs
        }

        if return_gradcam:
            last_conv = self.model.get_last_conv_layer()
            gradcam = GradCAM(self.model, last_conv)
            heatmap = gradcam.generate_heatmap(input_tensor, target_class=pred_idx)
            overlay = overlay_gradcam(img_pil, heatmap)
            result['gradcam_overlay'] = overlay
            result['gradcam_heatmap'] = heatmap

        return result
