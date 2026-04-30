from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import (
    CANDIDATE_COL,
    IMAGE_COL,
    REFERENCE_COL,
    ROW_ID_COL,
    config_needs_green,
    config_needs_image,
    json_dumps,
    normalize_input_dataframe,
    read_json,
    require_column,
    score_with_config,
    selected_english_metric_keys,
    selected_translators_and_kinds,
    write_json,
)


def merge_on_row_id(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    require_column(left, ROW_ID_COL)
    require_column(right, ROW_ID_COL)
    left = left.copy()
    right = right.copy()
    left[ROW_ID_COL] = left[ROW_ID_COL].astype(str)
    right[ROW_ID_COL] = right[ROW_ID_COL].astype(str)
    keep = [c for c in right.columns if c != ROW_ID_COL and c not in left.columns]
    return left.merge(right[[ROW_ID_COL] + keep], on=ROW_ID_COL, how="left")


def source_columns_from_config(config: Dict[str, Any]) -> Dict[str, str]:
    columns = config.get("columns") or {}
    required = ["candidate_col_source", "reference_col_source", "image_col_source"]
    missing = [k for k in required if not columns.get(k)]
    if missing:
        raise ValueError(f"Config does not contain source column names: {missing}")
    return {
        "candidate": columns["candidate_col_source"],
        "reference": columns["reference_col_source"],
        "image": columns["image_col_source"],
    }


def prepare_eval_csv(input_csv: str, tmp_dir: Path, config: Dict[str, Any]) -> Tuple[Path, pd.DataFrame]:
    original = pd.read_csv(input_csv)
    columns = source_columns_from_config(config)
    normalized, _ = normalize_input_dataframe(
        original,
        candidate_col=columns["candidate"],
        reference_col=columns["reference"],
        image_col=columns["image"],
        require_target=False,
    )
    base_csv = tmp_dir / "base.csv"
    normalized[[ROW_ID_COL, CANDIDATE_COL, REFERENCE_COL, IMAGE_COL]].to_csv(base_csv, index=False)
    return base_csv, normalized


def compute_required_features(args: argparse.Namespace, config: Dict[str, Any], base_csv: Path, tmp_dir: Path) -> pd.DataFrame:
    from calculate_english_text import calculate_english_text_metrics_for_csv
    from calculate_image_text import calculate_image_text_metrics_for_csv
    from calculate_russian_text import calculate_green_for_csv
    from translate_reports import translate_reports_for_csv

    features = pd.read_csv(base_csv)[[ROW_ID_COL]].copy()
    translators, kinds = selected_translators_and_kinds(config)
    needs_english = bool(selected_english_metric_keys(config))
    needs_image = config_needs_image(config)

    merged_for_english = None
    if needs_english or needs_image:
        translated_csv = tmp_dir / "translations.csv"
        translated = translate_reports_for_csv(
            input_csv=str(base_csv),
            output_csv=str(translated_csv),
            translators=translators,
            translation_kinds=kinds,
            qwen_model=config["models"].get("qwen_translator_model", "Qwen/Qwen2.5-7B-Instruct"),
            hy_mt_model=config["models"].get("hy_mt_model", "tencent/HY-MT1.5-7B"),
            translategemma_model=config["models"].get("translategemma_model", "google/translategemma-12b-it"),
            max_new_tokens=args.translation_max_new_tokens,
            device=args.device,
            gpu=args.gpu,
            per_translator_timeout=args.per_translator_timeout,
            verbose=args.verbose,
        )
        base = pd.read_csv(base_csv)
        merged_for_english = tmp_dir / "base_with_translations.csv"
        merge_on_row_id(base, translated).to_csv(merged_for_english, index=False)

    if config_needs_green(config):
        green_csv = tmp_dir / "green.csv"
        green = calculate_green_for_csv(
            input_csv=str(base_csv),
            output_csv=str(green_csv),
            model_name=config["models"].get("green_model", "Qwen/Qwen2.5-7B-Instruct"),
            batch_size=args.green_batch_size,
            max_new_tokens=args.green_max_new_tokens,
            device=args.device,
            gpu=args.gpu,
            verbose=args.verbose,
        )
        features = merge_on_row_id(features, green)

    if needs_english:
        english_csv = tmp_dir / "english.csv"
        english = calculate_english_text_metrics_for_csv(
            input_csv=str(merged_for_english),
            output_csv=str(english_csv),
            translators=translators,
            translation_kinds=kinds,
            methods=selected_english_metric_keys(config),
            cxrbert_model=config["models"].get("cxrbert_model", "microsoft/BiomedVLP-CXR-BERT-specialized"),
            biovilt_model=config["models"].get("biovilt_model", "microsoft/BiomedVLP-BioViL-T"),
            batch_size=args.text_batch_size,
            device=args.device,
            gpu=args.gpu,
            verbose=args.verbose,
        )
        features = merge_on_row_id(features, english)

    if needs_image:
        image_csv = tmp_dir / "image.csv"
        image = calculate_image_text_metrics_for_csv(
            input_csv=str(merged_for_english),
            output_csv=str(image_csv),
            translators=translators,
            translation_kinds=kinds,
            image_root=args.image_root,
            biovilt_model=config["models"].get("biovilt_model", "microsoft/BiomedVLP-BioViL-T"),
            batch_size=args.image_batch_size,
            device=args.device,
            gpu=args.gpu,
            verbose=args.verbose,
        )
        features = merge_on_row_id(features, image)

    return features


def to_json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def build_output(normalized: pd.DataFrame, features: pd.DataFrame, config: Dict[str, Any], scores: np.ndarray) -> Dict[str, Any]:
    merged = merge_on_row_id(normalized, features)
    selected = config["selected_features"]
    weights = config["weights"]
    entries: List[Dict[str, Any]] = []

    for i, row in merged.iterrows():
        metrics = []
        for spec, weight in zip(selected, weights):
            col = spec["feature_col"]
            metrics.append(
                {
                    "method_key": spec["method_key"],
                    "feature_col": col,
                    "translator": spec.get("translator"),
                    "translation_kind": spec.get("kind"),
                    "weight": float(weight),
                    "value": to_json_scalar(row.get(col)),
                }
            )
        entries.append(
            {
                CANDIDATE_COL: to_json_scalar(row[CANDIDATE_COL]),
                REFERENCE_COL: to_json_scalar(row[REFERENCE_COL]),
                IMAGE_COL: to_json_scalar(row[IMAGE_COL]),
                "quality_score": to_json_scalar(scores[i]),
                "metrics": metrics,
                "GREEN_analysis": to_json_scalar(row.get("ru_GREEN_Qwen_analysis")),
                "GREEN_errors_json": to_json_scalar(row.get("ru_GREEN_Qwen_errors_json")),
            }
        )

    return {
        "summary": {
            "quality_score_mean": float(np.nanmean(scores)) if len(scores) else None,
            "quality_score_std": float(np.nanstd(scores, ddof=1)) if len(scores) > 1 else 0.0,
        },
        "entry-wise": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--path-to-data-description-for-evaluation", required=True)
    parser.add_argument("--output", default="quality_scores.json")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--per-translator-timeout", type=int, default=3600)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--translation-max-new-tokens", type=int, default=256)
    parser.add_argument("--green-batch-size", type=int, default=1)
    parser.add_argument("--green-max-new-tokens", type=int, default=512)
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    os.environ["PYTHONPATH"] = str(project_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    config = read_json(args.config)

    with tempfile.TemporaryDirectory(prefix="metric_eval_") as d:
        tmp_dir = Path(d)
        base_csv, normalized = prepare_eval_csv(args.path_to_data_description_for_evaluation, tmp_dir, config)
        features = compute_required_features(args, config, base_csv, tmp_dir)
        scores = score_with_config(features, config)
        payload = build_output(normalized, features, config, scores)

    write_json(args.output, payload)
    print(json_dumps(payload["summary"]))


if __name__ == "__main__":
    main()
