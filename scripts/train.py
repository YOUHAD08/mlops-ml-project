import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from pathlib import Path
import yaml
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

from src.data import load_dataset
from src.features import build_numeric_preprocess
from src.model import build_model

def load_cfg(path="config/train.yaml"):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def save_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    cfg = load_cfg()
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    art_dir.mkdir(exist_ok=True)
    
    # Load data
    X, y = load_dataset(cfg)
    
    # Split data
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=float(cfg["split"]["test_size"]),
        random_state=int(cfg["split"]["random_state"]),
        stratify=y
    )
    
    # Build pipeline
    preprocess = build_numeric_preprocess()
    model = build_model(cfg)
    
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])
    
    # Train
    pipe.fit(Xtr, ytr)
    
    # Evaluate
    pred = pipe.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro"))
    
    # Save artifacts
    joblib.dump(pipe, art_dir / "model.joblib")
    json.dump(
        {"accuracy": acc, "f1_macro": f1}, 
        open(art_dir / "metrics.json", "w"), 
        indent=2
    )
    save_confusion_matrix(yte, pred, art_dir / "confusion_matrix.png")
    
    print("Training completed successfully!")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")

if __name__ == "__main__":
    main()