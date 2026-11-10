import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Generates visual explanations for model decisions on brain MRI scans.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap array [H, W] normalized between 0 and 1.
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()

        # Normalize heatmap between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

def overlay_gradcam(
    original_img_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap onto the original PIL MRI image.
    """
    img_rgb = np.array(original_img_pil.convert('RGB'))
    h, w, _ = img_rgb.shape

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap onto original image
    superimposed = np.float32(heatmap_colored) * alpha + np.float32(img_rgb) * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return superimposed

def plot_mri_with_gradcam(
    image_path: str,
    model: nn.Module,
    classes: list,
    device: torch.device,
    save_path: str = None
):
    """
    Plots original MRI scan alongside Grad-CAM heatmap and prediction confidence.
    """
    from src.dataset import NORM_MEAN, NORM_STD

    raw_img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Find last convolutional layer for EfficientNet
    last_conv = model.get_last_conv_layer() if hasattr(model, 'get_last_conv_layer') else model.backbone.features[-1]
    grad_cam = GradCAM(model, last_conv)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]
        confidence = probs[pred_idx] * 100

    heatmap = grad_cam.generate_heatmap(input_tensor, target_class=pred_idx)
    overlay = overlay_gradcam(raw_img, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_img)
    axes[0].set_title("Original MRI Scan", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(f"Grad-CAM Heatmap", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {pred_class} ({confidence:.1f}%)", fontsize=12, color='darkgreen')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"🔬 Saved Grad-CAM visualization to: {save_path}")
    plt.close()
