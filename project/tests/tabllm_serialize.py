"""
TabLLM-style serialization for Carte datasets.
Uses extended_meanings.json instead of hand-crafted feature descriptions.

Produces list-format serialization (TabLLM's best-performing format):
  - Feature description: value
  - Feature description: value
  ...

Usage:
    from tabllm_serialize import TabLLMSerializer
    serializer = TabLLMSerializer()
    text = serializer.serialize_row(X_row)
"""

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _format_value(value) -> str:
    """Format a cell value into a human-readable string."""
    if pd.isna(value):
        return "missing"
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if value == int(value):
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


class TabLLMSerializer:
    """Serializes tabular rows into natural language using column descriptions.

    Mirrors TabLLM's approach but uses extended_meanings.json for
    column descriptions instead of hand-crafted feature_names dicts.

    Parameters
    ----------
    extended_meanings_path : str, optional
        Path to extended_meanings.json
    format : str, default='list'
        'list' for bullet-point format (best in TabLLM paper),
        'text' for prose format
    """

    def __init__(
        self,
        extended_meanings_path: Optional[str] = None,
        format: str = "list",
    ):
        if extended_meanings_path is None:
            extended_meanings_path = os.path.join(
                os.path.dirname(__file__), "..", "rewrite_column", "extended_meanings.json"
            )

        with open(extended_meanings_path, "r") as f:
            self.meanings = json.load(f)

        if format not in ("list", "text"):
            raise ValueError(f"format must be 'list' or 'text', got '{format}'")
        self.format = format

    def get_description(self, dataset_name: str, column_name: str) -> str:
        """Get the human-readable description for a column."""
        if dataset_name in self.meanings:
            meaning = self.meanings[dataset_name].get(
                column_name, column_name.replace("_", " ").title()
            )
            if meaning[-1] == '.':
                meaning = meaning[:-1]
            return meaning
        return column_name.replace("_", " ").title()

    def serialize_row(
        self,
        dataset_name: str,
        row: pd.Series,
        exclude_columns=None,
        target_desc: str = None,
    ) -> str:
        """Serialize a single data row into natural language text.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset (must match key in extended_meanings.json)
        row : pd.Series
            A single row of feature values
        exclude_columns : set, optional
            Column names to exclude (e.g., target column)
        target_desc : str, optional
            Description of the prediction target (e.g., "Stars (rating, ≥4 is Good)")

        Returns
        -------
        str
            Serialized text in the chosen format
        """
        if exclude_columns is None:
            exclude_columns = set()

        parts = []
        for col in row.index:
            if col in exclude_columns:
                continue
            desc = self.get_description(dataset_name, col)
            value = _format_value(row[col])
            parts.append((desc, value))

        if self.format == "list":
            lines = [f"- {desc}: {value}" for desc, value in parts]
            if target_desc:
                lines.append(f"- {target_desc}:")
            return "\n".join(lines)
        else:
            sentences = [f"The {desc} is {value}." for desc, value in parts]
            if target_desc:
                sentences.append(f"What is the {target_desc}?")
            return " ".join(sentences)

    def serialize_dataset(
        self,
        dataset_name: str,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        target_desc: str = None,
    ) -> pd.DataFrame:
        """Serialize an entire dataset.

        Returns a DataFrame with columns:
        - 'text': serialized natural language text
        - 'label': target value (if y is provided)
        """
        exclude = set()
        if y is not None:
            exclude.add(y.name if hasattr(y, "name") and y.name else "target")

        texts = []
        for idx in X.index:
            text = self.serialize_row(dataset_name, X.loc[idx], exclude, target_desc)
            texts.append(text)

        result = pd.DataFrame({"text": texts})
        if y is not None:
            result["label"] = y.values if hasattr(y, "values") else list(y)
        return result


# ── Demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data import read_carte_classification_datasets

    serializer = TabLLMSerializer(format="list")
    datasets = read_carte_classification_datasets(max_samples=3)

    for ds_name, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  |  Target: {y.name}")
        print(f"{'='*60}")
        for i in range(min(2, len(X))):
            text = serializer.serialize_row(ds_name, X.iloc[i])
            print(f"\n--- Row {i} ---")
            print(text)
            print(f"Label: {y.iloc[i]}")
