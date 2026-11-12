#!/usr/bin/env python3
"""
Compare Corpus Reports Module

Collects the per-corpus reports generated in results/corpus_analysis
(frequency_corpus_report.csv, frequency_entities_report.csv and
linguistic_analysis_report.csv) and merges each one into a single CSV with the
results for train, test and every model/strategy, both in absolute terms and as
the difference with respect to train (value - train).
"""

# --- Local Imports ---
from ..imports import os, pd

RESULTS_DIR = "results/corpus_analysis"
REFERENCE = "train"
CSV_SEP = ";"


def parse_corpus_name(dirname: str) -> tuple[str, str]:
    """
    Splits a results directory name into (model, strategy).

    Args:
        dirname: Directory name, e.g. "llama32-3b_GSA#2", "train" or "test".

    Returns:
        Tuple (model, strategy). train/test are labelled as model "Original".
    """
    if dirname in ("train", "test"):
        return "Original", dirname
    model, _, strategy = dirname.rpartition("_")
    return model, strategy


def to_number(value) -> float:
    """Converts values such as '36,949', '0.1000' or '-' to float (NaN if not numeric)."""
    return pd.to_numeric(str(value).replace(",", ""), errors="coerce")


def list_corpora(results_dir: str) -> list[str]:
    """Returns corpus directories ordered as train, test, then models/strategies alphabetically."""
    names = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    first = [d for d in ("train", "test") if d in names]
    return first + sorted(d for d in names if d not in first)


def read_reports(results_dir: str, filename: str) -> pd.DataFrame:
    """
    Reads `filename` from every corpus directory and stacks them with Model/Strategy columns.

    Args:
        results_dir: Directory containing one sub-directory per corpus.
        filename: Report CSV name to collect.

    Returns:
        Long DataFrame with Model and Strategy as the first two columns.
    """
    frames = []
    for dirname in list_corpora(results_dir):
        path = os.path.join(results_dir, dirname, filename)
        if not os.path.exists(path):
            print(f"[WARNING] Missing {path}, skipping.")
            continue
        df = pd.read_csv(path, sep=CSV_SEP, dtype=str)
        model, strategy = parse_corpus_name(dirname)
        df.insert(0, "Strategy", strategy)
        df.insert(0, "Model", model)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_differences(df: pd.DataFrame, keys: list[str], value_cols: list[str]) -> pd.DataFrame:
    """
    Converts value columns to numbers and appends `<col>_diff_vs_train` columns.

    Args:
        df: Wide DataFrame with Model, Strategy, key columns and value columns.
        keys: Columns that identify a row inside a corpus (empty for one row per corpus).
        value_cols: Numeric columns for which to compute the difference with train.

    Returns:
        DataFrame with absolute values followed by the differences with train.
    """
    df = df.copy()
    df[value_cols] = df[value_cols].apply(lambda s: s.map(to_number))

    train = df[(df["Model"] == "Original") & (df["Strategy"] == REFERENCE)]
    if keys:
        diff = df.merge(train[keys + value_cols], on=keys, how="left", suffixes=("", "_train"))
        train_values = diff[[f"{c}_train" for c in value_cols]].to_numpy()
    else:
        train_values = train[value_cols].iloc[[0] * len(df)].to_numpy()

    for i, col in enumerate(value_cols):
        df[f"{col}_diff_vs_train"] = (df[col].to_numpy() - train_values[:, i]).round(4)
    return df


def compare_corpus_report(results_dir: str) -> pd.DataFrame:
    """frequency_corpus_report.csv: one row per corpus, one column per metric."""
    long = read_reports(results_dir, "frequency_corpus_report.csv")
    wide = long.pivot(index=["Model", "Strategy"], columns="Metric", values="Value")
    wide = wide[list(dict.fromkeys(long["Metric"]))].reset_index()
    return add_differences(wide, [], [c for c in wide.columns if c not in ("Model", "Strategy")])


def compare_entities_report(results_dir: str) -> pd.DataFrame:
    """frequency_entities_report.csv: one row per corpus and entity type."""
    df = read_reports(results_dir, "frequency_entities_report.csv")
    cols = ["Model", "Strategy", "Entity Type"] + [
        c for c in df.columns if c not in ("Model", "Strategy", "Entity Type")]
    df = df[cols]
    return add_differences(df, ["Entity Type"], cols[3:])


def compare_linguistic_report(results_dir: str) -> pd.DataFrame:
    """linguistic_analysis_report.csv: one row per corpus, Absolute/Relative columns per metric."""
    long = read_reports(results_dir, "linguistic_analysis_report.csv")
    wide = long.pivot(index=["Model", "Strategy"], columns="Metric", values=["Absolute", "Relative"])
    order = list(dict.fromkeys(long["Metric"]))
    wide = wide.reindex(columns=[(kind, m) for m in order for kind in ("Absolute", "Relative")])
    wide.columns = [f"{m} [{kind}]" for kind, m in wide.columns]
    wide = wide.reset_index()
    return add_differences(wide, [], [c for c in wide.columns if c not in ("Model", "Strategy")])


def sort_corpora(df: pd.DataFrame, results_dir: str) -> pd.DataFrame:
    """Orders rows as train, test and then models/strategies (keeping entity order inside each)."""
    order = {parse_corpus_name(d): i for i, d in enumerate(list_corpora(results_dir))}
    rank = [order[(m, s)] for m, s in zip(df["Model"], df["Strategy"])]
    return df.assign(_rank=rank).sort_values("_rank", kind="stable").drop(columns="_rank")


def compare_reports(results_dir: str = RESULTS_DIR) -> None:
    """
    Builds the three comparison CSVs and saves them in `results_dir`.

    Args:
        results_dir: Directory with the per-corpus results (also used as output directory).
    """
    outputs = {
        "comparison_frequency_corpus_report.csv": compare_corpus_report,
        "comparison_frequency_entities_report.csv": compare_entities_report,
        "comparison_linguistic_analysis_report.csv": compare_linguistic_report,
    }
    for filename, builder in outputs.items():
        df = sort_corpora(builder(results_dir), results_dir)
        path = os.path.join(results_dir, filename)
        df.to_csv(path, index=False, sep=CSV_SEP)
        print(f"Saved {path} ({len(df)} rows)")
