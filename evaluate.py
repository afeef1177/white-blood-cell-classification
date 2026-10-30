import argparse, os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.data import create_dataloaders
from src.model import build_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default="outputs/best_model.pth")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    loaders, classes = create_dataloaders(args.data_dir, args.batch_size)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if ckpt["classes"] != classes: raise ValueError("Class order mismatch between checkpoint and dataset")
    model = build_model(ckpt["model_name"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loaders["test"]:
            preds = model(images.to(device)).argmax(1).cpu().tolist()
            y_pred.extend(preds); y_true.extend(labels.tolist())
    print(classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    os.makedirs("outputs", exist_ok=True)
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(xticks_rotation=45, values_format="d")
    plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=200)
    print("Saved outputs/confusion_matrix.png")

if __name__ == "__main__": main()
