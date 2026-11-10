import os
import sys
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def report_progress(
    pct: float,
    train_loss: float,
    train_acc: float,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    end: str = "\r"
) -> None:
    """
    Displays clean, real-time training progress strictly limited to:
    - Progress percentage and Remaining percentage
    - Loss (Train / Val)
    - Accuracy (Train / Val)
    """
    rem_pct = max(0.0, 100.0 - pct)
    filled = min(20, max(0, int(pct / 5)))
    bar = "█" * filled + "░" * (20 - filled)
    
    if val_loss is not None and val_acc is not None:
        metrics = f"الفقد: {train_loss:.4f} (تحقق: {val_loss:.4f}) | الدقة: {train_acc * 100:5.2f}% (تحقق: {val_acc * 100:5.2f}%)"
    else:
        metrics = f"الفقد: {train_loss:.4f} | الدقة: {train_acc * 100:5.2f}%"
        
    line = f"[{bar}] التقدم: {pct:5.1f}% | المتبقي: {rem_pct:5.1f}% | {metrics}"
    print(f"\r{line:<110}", end=end, flush=True)

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    global_step_tracker: Optional[Dict[str, int]] = None,
    total_steps: Optional[int] = None
) -> Tuple[float, float]:
    """
    Trains model for one epoch using Automatic Mixed Precision (AMP) and live in-place progress counter.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == 'cuda'
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        if global_step_tracker is not None and total_steps:
            global_step_tracker['step'] += 1
            pct = (global_step_tracker['step'] / total_steps) * 100
            report_progress(pct, running_loss / total, correct / total, end="\r")

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates model on validation/test set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == 'cuda'
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    return running_loss / total, correct / total

def train_two_stages(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    stage1_epochs: int = 4,
    stage2_epochs: int = 16,
    save_path: str = "artifacts/best_brain_tumor_model.pth",
    device: Optional[torch.device] = None,
    single_line_display: bool = False
) -> Dict[str, List[float]]:
    """
    Two-Stage Transfer Learning with clean output restricted strictly to:
    - Percentage counter (Completed % & Remaining %)
    - Loss
    - Accuracy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')

    total_epochs = stage1_epochs + stage2_epochs
    total_steps = total_epochs * len(train_loader)
    step_tracker = {'step': 0}

    # ================= Stage 1: Warmup =================
    if hasattr(model, 'freeze_backbone'):
        model.freeze_backbone(freeze=True)

    optimizer_s1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, stage1_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_s1, scaler, device,
            global_step_tracker=step_tracker, total_steps=total_steps
        )
        v_loss, v_acc = evaluate_epoch(model, val_loader, criterion, device)
        
        pct = (step_tracker['step'] / total_steps) * 100
        epoch_end = "\r" if single_line_display else "\n"
        report_progress(pct, tr_loss, tr_acc, v_loss, v_acc, end=epoch_end)

    # ================= Stage 2: End-to-End Fine-Tuning =================
    if hasattr(model, 'freeze_backbone'):
        model.freeze_backbone(freeze=False)

    optimizer_s2 = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=stage2_epochs, eta_min=1e-6)

    for epoch in range(1, stage2_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_s2, scaler, device,
            global_step_tracker=step_tracker, total_steps=total_steps
        )
        v_loss, v_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler_s2.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        pct = (step_tracker['step'] / total_steps) * 100
        epoch_end = "\r" if single_line_display else "\n"
        report_progress(pct, tr_loss, tr_acc, v_loss, v_acc, end=epoch_end)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save({
                'epoch': stage1_epochs + epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': v_loss,
                'val_acc': v_acc
            }, save_path)

    return history