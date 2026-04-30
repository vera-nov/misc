from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils import (
    CANDIDATE_COL,
    IMAGE_COL,
    REFERENCE_COL,
    ROW_ID_COL,
    TARGET_COL,
    config_needs_green,
    config_needs_image,
    normalize_input_dataframe,
    read_json,
    run_module_function_in_process,
    score_with_config,
    selected_english_metric_keys,
    selected_translators_and_kinds,
    split_csv_arg,
)
from translate_reports import DEFAULT_HY_MT_MODEL, DEFAULT_QWEN_MODEL, DEFAULT_TRANSLATEGEMMA_MODEL
from calculate_english_text import DEFAULT_BIOVILT_MODEL, DEFAULT_CXRBERT_MODEL
from calculate_russian_text import DEFAULT_MODEL as DEFAULT_GREEN_MODEL


def merge_on_row_id(left: pd.DataFrame, right_path: str | Path, row_id_col: str = ROW_ID_COL) -> pd.DataFrame:
    left = left.copy()
    right = pd.read_csv(right_path, dtype={row_id_col: str})

    if row_id_col not in left.columns:
        raise ValueError(f"Left dataframe has no merge key column '{row_id_col}'.")
    if row_id_col not in right.columns:
        raise ValueError(f"Right dataframe '{right_path}' has no merge key column '{row_id_col}'.")

    left[row_id_col] = left[row_id_col].astype(str)
    right[row_id_col] = right[row_id_col].astype(str)

    drop = [c for c in right.columns if c != row_id_col and c in left.columns]
    if drop:
        left = left.drop(columns=drop)

    return left.merge(right, on=row_id_col, how="left")


def prepare_eval_csv(
    input_csv: str,
    tmp_path: str,
    config: Dict[str, Any],
    candidate_col: Optional[str],
    reference_col: Optional[str],
    image_col: Optional[str],
    row_id_col: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    original = pd.read_csv(input_csv)
    mapping = config.get("input_mapping", {})
    normalized, new_mapping = normalize_input_dataframe(
        original,
        candidate_col=candidate_col or mapping.get("candidate_col_source"),
        reference_col=reference_col or mapping.get("reference_col_source"),
        image_col=image_col or mapping.get("image_col_source"),
        target_col=None,
        require_target=False,
        row_id_col=row_id_col or mapping.get("row_id_col_source"),
    )
    normalized.to_csv(tmp_path, index=False)
    return original, normalized, new_mapping


def compute_required_features(
    base_csv: str,
    normalized_df: pd.DataFrame,
    config: Dict[str, Any],
    tmp_dir: str,
    gpu: str,
    device: str,
    timeout: Optional[int],
    image_root: Optional[str],
    green_batch_size: int,
    text_batch_size: int,
    image_batch_size: int,
    max_new_tokens_translation: int,
    max_new_tokens_green: int,
    verbose: bool,
) -> pd.DataFrame:
    tmp = Path(tmp_dir)
    translators, kinds = selected_translators_and_kinds(config)
    feature_df = normalized_df[[ROW_ID_COL]].copy()
    model_names = config.get("model_names", {})
    qwen_model = model_names.get("qwen", DEFAULT_QWEN_MODEL)
    hy_model = model_names.get("hy_mt", DEFAULT_HY_MT_MODEL)
    gemma_model = model_names.get("translategemma", DEFAULT_TRANSLATEGEMMA_MODEL)
    green_model = model_names.get("green", DEFAULT_GREEN_MODEL)
    cxrbert_model = model_names.get("cxrbert", DEFAULT_CXRBERT_MODEL)
    biovilt_model = model_names.get("biovilt", DEFAULT_BIOVILT_MODEL)

    translated_csv: Optional[Path] = None
    if translators:
        translation_csv = tmp / "translations.csv"
        if verbose:
            print("[evaluate] translations in isolated process", flush=True)
        run_module_function_in_process(
            "translate_reports",
            "translate_reports_for_csv",
            {
                "input_csv": base_csv,
                "output_csv": str(translation_csv),
                "candidate_col": CANDIDATE_COL,
                "reference_col": REFERENCE_COL,
                "row_id_col": ROW_ID_COL,
                "translators": translators,
                "kinds": kinds,
                "model_names": {"qwen": qwen_model, "hy_mt": hy_model, "translategemma": gemma_model},
                "gpu": gpu,
                "device": device,
                "max_new_tokens": max_new_tokens_translation,
            },
            timeout=timeout,
        )
        translated_df = merge_on_row_id(normalized_df, translation_csv, row_id_col=ROW_ID_COL)
        translated_csv = tmp / "base_plus_translations.csv"
        translated_df.to_csv(translated_csv, index=False)

    if config_needs_green(config):
        green_csv = tmp / "russian_green.csv"
        if verbose:
            print("[evaluate] Russian GREEN/Qwen in isolated process", flush=True)
        run_module_function_in_process(
            "calculate_russian_text",
            "calculate_green_for_csv",
            {
                "input_csv": base_csv,
                "output_csv": str(green_csv),
                "candidate_col": CANDIDATE_COL,
                "reference_col": REFERENCE_COL,
                "row_id_col": ROW_ID_COL,
                "model_name": green_model,
                "batch_size": green_batch_size,
                "gpu": gpu,
                "device": device,
                "max_new_tokens": max_new_tokens_green,
            },
            timeout=timeout,
        )
        feature_df = merge_on_row_id(feature_df, green_csv)

    english_methods = selected_english_metric_keys(config)
    if english_methods:
        if translated_csv is None:
            raise RuntimeError("Config needs English text metrics, but no translators were selected.")
        text_csv = tmp / "english_text_metrics.csv"
        if verbose:
            print("[evaluate] English text metrics in isolated process", flush=True)
        run_module_function_in_process(
            "calculate_english_text",
            "calculate_english_text_metrics_for_csv",
            {
                "input_csv": str(translated_csv),
                "output_csv": str(text_csv),
                "row_id_col": ROW_ID_COL,
                "translators": translators,
                "kinds": kinds,
                "methods": english_methods,
                "cxrbert_model": cxrbert_model,
                "biovilt_model": biovilt_model,
                "batch_size": text_batch_size,
                "gpu": gpu,
                "device": device,
            },
            timeout=timeout,
        )
        feature_df = merge_on_row_id(feature_df, text_csv)

    if config_needs_image(config):
        if translated_csv is None:
            raise RuntimeError("Config needs image-text metric, but no translators were selected.")
        image_csv = tmp / "image_text_metrics.csv"
        if verbose:
            print("[evaluate] BioViL-T image-text metric in isolated process", flush=True)
        run_module_function_in_process(
            "calculate_image_text",
            "calculate_image_text_metrics_for_csv",
            {
                "input_csv": str(translated_csv),
                "output_csv": str(image_csv),
                "image_col": IMAGE_COL,
                "row_id_col": ROW_ID_COL,
                "translators": translators,
                "kinds": kinds,
                "image_root": image_root,
                "biovilt_model": biovilt_model,
                "batch_size": image_batch_size,
                "gpu": gpu,
                "device": device,
            },
            timeout=timeout,
        )
        feature_df = merge_on_row_id(feature_df, image_csv)

    return feature_df


def save_output(output_path: str | Path, original_df: pd.DataFrame, result_df: pd.DataFrame, summary: Dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        payload = {"summary": summary, "rows": result_df.to_dict(orient="records")}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
            for rec in result_df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        # Single output file: add summary as constant columns.
        out = result_df.copy()
        out["quality_score_mean"] = summary["quality_score_mean"]
        out["quality_score_std"] = summary["quality_score_std"]
        out.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate reports using saved radiology quality ensemble config.")
    p.add_argument("--config", required=True, help="Path to config JSON produced by build_method.py")
    p.add_argument("--path-to-data-description-for-evaluation", "--path_to_data_description_for_evaluation", dest="input_csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--candidate-col", required=True)
    p.add_argument("--reference-col", required=True)
    p.add_argument("--image-col", required=True)
    p.add_argument("--row-id-col", default=None)
    p.add_argument("--gpu", default="0")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--image-root", default=None, help="Root for relative image paths. Default: config image_root or input CSV directory.")
    p.add_argument("--green-batch-size", type=int, default=1)
    p.add_argument("--text-batch-size", type=int, default=16)
    p.add_argument("--image-batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens-translation", type=int, default=256)
    p.add_argument("--max-new-tokens-green", type=int, default=768)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    image_root = args.image_root or config.get("image_root") or str(input_csv.parent)

    with tempfile.TemporaryDirectory(prefix="radiology_quality_eval_") as td:
        base_csv = str(Path(td) / "base.csv")
        original_df, normalized_df, mapping = prepare_eval_csv(
            str(input_csv),
            base_csv,
            config,
            args.candidate_col,
            args.reference_col,
            args.image_col,
            args.row_id_col,
        )
        feature_df = compute_required_features(
            base_csv=base_csv,
            normalized_df=normalized_df,
            config=config,
            tmp_dir=td,
            gpu=args.gpu,
            device=args.device,
            timeout=args.timeout,
            image_root=image_root,
            green_batch_size=args.green_batch_size,
            text_batch_size=args.text_batch_size,
            image_batch_size=args.image_batch_size,
            max_new_tokens_translation=args.max_new_tokens_translation,
            max_new_tokens_green=args.max_new_tokens_green,
            verbose=args.verbose,
        )
        scores = score_with_config(feature_df, config)
        selected_cols = [s["feature_col"] for s in config["selected_features"]]
        result = normalized_df.copy()
        for col in selected_cols:
            if col in feature_df.columns:
                result[col] = feature_df[col]
        result["quality_score"] = scores
        # Add explainable error list if selected / available.
        for spec in config.get("selected_features", []):
            errors_col = spec.get("errors_col")
            if errors_col and errors_col in feature_df.columns:
                result["llm_errors_json"] = feature_df[errors_col]
        summary = {
            "n_rows": int(len(result)),
            "quality_score_mean": float(np.nanmean(scores)),
            "quality_score_std": float(np.nanstd(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "selected_features": config["selected_features"],
            "weights": config["weights"],
            "input_mapping_used": mapping,
        }
        save_output(args.output, original_df, result, summary)
    print(f"Saved evaluation: {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
