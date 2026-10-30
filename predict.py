import argparse, torch
from PIL import Image
from src.data import build_transforms
from src.model import build_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default="outputs/best_model.pth")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.to(device).eval()
    _, eval_tf = build_transforms(ckpt.get("image_size", 224))
    x = eval_tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad(): probs = torch.softmax(model(x), dim=1)[0]
    score, idx = probs.max(0)
    print(f"Predicted class: {ckpt['classes'][idx.item()]}")
    print(f"Confidence: {score.item():.4f} (not a clinical diagnosis)")

if __name__ == "__main__": main()
