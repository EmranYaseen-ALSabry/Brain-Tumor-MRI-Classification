import os
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support
)

@torch.no_grad()
def get_predictions(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """
    Runs model on test/val dataloader and returns true labels, predictions, and softmax probabilities.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def evaluate_model_performance(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    classes: List[str],
    device: torch.device,
    output_dir: str = "artifacts"
) -> Dict:
    """
    Evaluates model and generates medical classification reports and visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
    except Exception:
        auc = 0.0

    print("=" * 60)
    print("🏥 MEDICAL EVALUATION METRICS REPORT")
    print("=" * 60)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Weighted Recall:    {rec * 100:.2f}% (Sensitivity)")
    print(f"Weighted Precision: {prec * 100:.2f}%")
    print(f"Weighted F1-Score:  {f1 * 100:.2f}%")
    if auc > 0:
        print(f"Weighted ROC-AUC:   {auc * 100:.2f}%")
    print("-" * 60)
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title("Brain Tumor MRI Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"📊 Confusion Matrix saved to: {cm_path}")

    return {
        'accuracy': acc,
        'recall': rec,
        'precision': prec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

def plot_training_history(history: Dict[str, List[float]], output_dir: str = "artifacts"):
    """
    Plots and saves loss and accuracy curves across training epochs.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    plt.title('Training & Validation Loss', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Accuracy')
    plt.plot(epochs, [a * 100 for a in history['val_acc']], 'r--s', label='Val Accuracy')
    plt.title('Training & Validation Accuracy (%)', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    history_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(history_path, dpi=300)
    plt.close()
    print(f"📈 Training History plot saved to: {history_path}")
