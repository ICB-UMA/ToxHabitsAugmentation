from typing import Dict, Tuple
import pandas as pd


def compute_span_prf_global(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute micro Precision / Recall / F1 over exact span matches.

    Matching key:
      (filename, start_span, end_span, label)
    """

    pred_set = set(
        zip(
            pred_df["filename"].astype(str),
            pred_df["start_span"].astype(int),
            pred_df["end_span"].astype(int),
            pred_df["label"].astype(str),
        )
    )

    gold_set = set(
        zip(
            gold_df["filename"].astype(str),
            gold_df["start_span"].astype(int),
            gold_df["end_span"].astype(int),
            gold_df["label"].astype(str),
        )
    )

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_span_prf_by_label(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Precision / Recall / F1 per entity label (strict span match).

    Returns a DataFrame with:
      label | tp | fp | fn | precision | recall | f1
    """

    labels = sorted(set(gold_df["label"].unique()) | set(pred_df["label"].unique()))
    rows = []

    for lab in labels:
        pred_lab = pred_df[pred_df["label"] == lab]
        gold_lab = gold_df[gold_df["label"] == lab]

        pred_set = set(
            zip(
                pred_lab["filename"].astype(str),
                pred_lab["start_span"].astype(int),
                pred_lab["end_span"].astype(int),
            )
        )

        gold_set = set(
            zip(
                gold_lab["filename"].astype(str),
                gold_lab["start_span"].astype(int),
                gold_lab["end_span"].astype(int),
            )
        )

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "label": lab,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)
