import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, targets_all, preds_all = 0.0, [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            targets_all.extend(targets.detach().cpu().tolist())
            preds_all.extend(logits.argmax(1).detach().cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_all, preds_all, average="macro", zero_division=0
    )
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": float(accuracy_score(targets_all, preds_all)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
