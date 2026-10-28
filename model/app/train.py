import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from app.corpus import build
from app.text import preprocess_str

SEED = 42
ROOT = Path(__file__).resolve().parent.parent


def scores(y, p) -> dict:
    return {
        "accuracy": round(accuracy_score(y, p), 4),
        "precision": round(precision_score(y, p), 4),
        "recall": round(recall_score(y, p), 4),
        "f1": round(f1_score(y, p), 4),
    }


def train(out_dir: Path) -> dict:
    df = build()
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    stress = pd.read_csv(ROOT / "data" / "stress_test.csv")

    # символьные n-граммы устойчивы к опечаткам и подмене букв, поэтому выбраны как итоговая модель
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=3000, C=10),
    )
    pipe.fit(tr.text.map(preprocess_str), tr.label)

    version = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "version": version,
        "train_size": len(tr),
        "test": scores(te.label, pipe.predict(te.text.map(preprocess_str))),
        "stress": scores(stress.label, pipe.predict(stress.text.map(preprocess_str))),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "version": version}, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts")
    print(json.dumps(train(parser.parse_args().out), indent=2))
