"""
Step 1: Serialize all Carte datasets into JSONL files with SEMANTIC labels.

Uses label_semantics.py to map each dataset's 0/1 labels back to their
original meaning (e.g., ramen: 1→"Good", 0→"Average").

Output structure:
    serialized_data/
        dataset_name/
            train.jsonl    ← {"text": "...", "label": "Good"}
            test.jsonl
"""

import json
import os
import sys
from sklearn.model_selection import train_test_split

root_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root_dir)

from data import read_carte_classification_datasets
from tabllm_serialize import TabLLMSerializer
from label_semantics import LABEL_SEMANTICS


OUTPUT_DIR = os.path.join(root_dir, "serialized_data")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    serializer = TabLLMSerializer(format="list")

    datasets = read_carte_classification_datasets(max_samples=None)

    for ds_name, (X, y) in datasets.items():
        ds_dir = os.path.join(OUTPUT_DIR, ds_name)
        os.makedirs(ds_dir, exist_ok=True)

        # Get semantic labels for this dataset
        semantics = LABEL_SEMANTICS.get(ds_name, None)
        if semantics is None:
            print(f"⚠️  {ds_name}: no semantics found, using Yes/No fallback")
            label_1, label_0 = "Yes", "No"
            target_desc = "target"
        else:
            label_1 = semantics["answer_1"]
            label_0 = semantics["answer_0"]
            target_col = semantics["target_col"]
            threshold = semantics.get("threshold", "N/A")
            target_desc = f"{target_col} ({label_1} if ≥{threshold})" if threshold else f"{target_col} ({label_1} or {label_0})"
            print(f"📋 {ds_name}: 1→'{label_1}', 0→'{label_0}' "
                  f"(target: {target_col}, threshold: {threshold})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        for split_name, (X_split, y_split) in [
            ("train", (X_train, y_train)),
            ("test", (X_test, y_test)),
        ]:
            df = serializer.serialize_dataset(ds_name, X_split, y_split, target_desc=target_desc)

            # Map numeric labels to SEMANTIC text
            # Use .values to avoid pandas index alignment issues (Bug fix: NaN labels)
            unique_labels = sorted(y_split.unique())
            if len(unique_labels) == 2:
                label_map = {unique_labels[0]: label_0, unique_labels[1]: label_1}
            else:
                label_map = {lbl: str(lbl) for lbl in unique_labels}
            df["label"] = y_split.map(label_map).values  # ← .values fixes NaN

            filepath = os.path.join(ds_dir, f"{split_name}.jsonl")
            with open(filepath, "w") as f:
                for _, row in df.iterrows():
                    f.write(json.dumps({"text": row["text"], "label": row["label"]}) + "\n")

            print(f"   ✅ {split_name}.jsonl  —  {len(df)} samples")

    print(f"\n📁 All saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
