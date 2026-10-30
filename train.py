import argparse, json, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from src.data import create_dataloaders
from src.engine import run_epoch
from src.model import build_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    loaders, classes = create_dataloaders(args.data_dir, args.batch_size, args.image_size, args.num_workers)
    print(f"Device={device}; classes={classes}")
    model = build_model(args.model, len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    os.makedirs("outputs", exist_ok=True)
    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_m = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_m["macro_f1"])
        history.append({"epoch": epoch, "train": train_m, "validation": val_m})
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_m['loss']:.4f} | "
              f"val loss {val_m['loss']:.4f} | val acc {val_m['accuracy']:.4f} | "
              f"val macro-F1 {val_m['macro_f1']:.4f}")
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            torch.save({"model_name": args.model, "model_state_dict": model.state_dict(),
                        "classes": classes, "image_size": args.image_size,
                        "best_val_macro_f1": best_f1}, "outputs/best_model.pth")
    with open("outputs/training_history.json", "w") as f: json.dump(history, f, indent=2)
    print(f"Finished. Best validation macro-F1={best_f1:.4f}")

if __name__ == "__main__": main()
