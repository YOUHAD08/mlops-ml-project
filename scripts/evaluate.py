import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from pathlib import Path
import joblib
import yaml
from sklearn.metrics import classification_report

from src.data import load_dataset

def load_cfg(path="config/train.yaml"):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def main():
    cfg = load_cfg()
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    
    # Load trained model
    model = joblib.load(art_dir / "model.joblib")
    
    # Load data
    X, y = load_dataset(cfg)
    
    # Predict
    pred = model.predict(X)
    
    # Generate detailed report
    report = classification_report(y, pred, output_dict=True)
    
    # Save report
    json.dump(report, open(art_dir / "report.json", "w"), indent=2)
    
    print("Evaluation completed successfully!")
    print(f"Report saved to: artifacts/report.json")

if __name__ == "__main__":
    main()