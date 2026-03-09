"""
Inference Script – Predict fish species from image(s).

Usage
-----
    # Single image
    python predict.py --image path/to/fish.jpg --model outputs/best_model.h5

    # Directory of images
    python predict.py --image_dir path/to/images/ --model outputs/best_model.h5

    # With custom top-k
    python predict.py --image fish.jpg --model outputs/best_model.h5 --top_k 3
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser(description="Fish Classifier – Inference")
    parser.add_argument("--model",       type=str, required=True,
                        help="Path to the saved .h5 model")
    parser.add_argument("--classes",     type=str, default="outputs/class_names.json",
                        help="Path to class_names.json")
    parser.add_argument("--image",       type=str, default=None,
                        help="Single image path")
    parser.add_argument("--image_dir",   type=str, default=None,
                        help="Directory of images to classify")
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--top_k",       type=int, default=3,
                        help="Show top-K predictions")
    return parser.parse_args()


def load_and_preprocess(image_path: str, img_size: int = 224) -> np.ndarray:
    """Load an image file and return a normalised batch tensor."""
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_image(model, image_path: str, class_names: list, img_size: int = 224, top_k: int = 3):
    """Run inference on a single image and print results."""
    tensor = load_and_preprocess(image_path, img_size)
    probs  = model.predict(tensor, verbose=0)[0]

    top_idx     = np.argsort(probs)[::-1][:top_k]
    top_classes = [class_names[i] for i in top_idx]
    top_probs   = [float(probs[i]) for i in top_idx]

    print(f"\nImage : {os.path.basename(image_path)}")
    print("-" * 40)
    for rank, (cls, prob) in enumerate(zip(top_classes, top_probs), 1):
        bar = "█" * int(prob * 30)
        print(f"  #{rank}  {cls:<25}  {prob*100:6.2f}%  {bar}")

    return top_classes[0], top_probs[0]


def main():
    args = parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[Predict] Loading model from {args.model} …")
    model = tf.keras.models.load_model(args.model)
    print("[Predict] Model loaded ✓")

    # ── Load class names ──────────────────────────────────────────────────────
    with open(args.classes) as f:
        class_names = json.load(f)
    print(f"[Predict] {len(class_names)} classes loaded ✓")

    # ── Collect image paths ───────────────────────────────────────────────────
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for fname in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(fname)[1].lower() in valid_ext:
                image_paths.append(os.path.join(args.image_dir, fname))
    else:
        print("[Predict] Please provide --image or --image_dir")
        sys.exit(1)

    print(f"\n[Predict] Running inference on {len(image_paths)} image(s) …\n")

    # ── Run predictions ───────────────────────────────────────────────────────
    results = []
    for img_path in image_paths:
        pred_class, confidence = predict_image(
            model, img_path, class_names, args.img_size, args.top_k
        )
        results.append({"image": img_path, "predicted_class": pred_class,
                        "confidence": round(confidence, 4)})

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = "prediction_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Predict] Results saved → {out_path}")


if __name__ == "__main__":
    main()
