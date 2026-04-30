from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from utils import (
    BASIC_METHODS,
    CANDIDATE_COL,
    IMAGE_COL,
    REFERENCE_COL,
    ROW_ID_COL,
    TARGET_COL,
    normalize_input_dataframe,
    read_json,
    run_module_function_in_process,
    select_and_fit_final_config,
    split_csv_arg,
    write_json,
)
from translate_reports import DEFAULT_HY_MT_MODEL, DEFAULT_QWEN_MODEL, DEFAULT_TRANSLATEGEMMA_MODEL
from calculate_english_text import ALL_ENGLISH_TEXT_METHODS, DEFAULT_BIOVILT_MODEL, DEFAULT_CXRBERT_MODEL
from calculate_russian_text import DEFAULT_MODEL as DEFAULT_GREEN_MODEL


def merge_on_row_id(left: pd.DataFrame, right_path: str | Path, row_id_col: str = ROW_ID_COL) -> pd.DataFrame:
    right = pd.read_csv(right_path)
    if row_id_col not in right.columns:
        raise ValueError(f"Worker output {right_path} has no {row_id_col} column")
    drop = [c for c in right.columns if c != row_id_col and c in left.columns]
    if drop:
        left = left.drop(columns=drop)
    return left.merge(right, on=row_id_col, how="left")


def prepare_base_csv(
    input_csv: str,
    tmp_path: str,
    candidate_col: Optional[str],
    reference_col: Optional[str],
    image_col: Optional[str],
    target_col: Optional[str],
    row_id_col: Optional[str],
) -> Dict[str, Any]:
    original = pd.read_csv(input_csv)
    normalized, mapping = normalize_input_dataframe(
        original,
        candidate_col=candidate_col,
        reference_col=reference_col,
        image_col=image_col,
        target_col=target_col,
        require_target=True,
        row_id_col=row_id_col,
    )
    normalized.to_csv(tmp_path, index=False)
    return {"mapping": mapping, "n_rows": int(len(normalized)), "columns": list(normalized.columns)}


def compute_all_features(
    base_csv: str,
    tmp_dir: str,
    translators: Sequence[str],
    kinds: Sequence[str],
    gpu: str,
    device: str,
    timeout: Optional[int],
    model_names: Dict[str, str],
    green_batch_size: int,
    text_batch_size: int,
    image_batch_size: int,
    image_root: Optional[str],
    max_new_tokens_translation: int,
    max_new_tokens_green: int,
    verbose: bool,
) -> pd.DataFrame:
    tmp = Path(tmp_dir)
    base_df = pd.read_csv(base_csv)

    translation_csv = tmp / "translations.csv"
    if verbose:
        print("[build] translations in isolated process", flush=True)
    run_module_function_in_process(
        "translate_reports",
        "translate_reports_for_csv",
        {
            "input_csv": base_csv,
            "output_csv": str(translation_csv),
            "candidate_col": CANDIDATE_COL,
            "reference_col": REFERENCE_COL,
            "row_id_col": ROW_ID_COL,
            "translators": list(translators),
            "kinds": list(kinds),
            "model_names": {"qwen": model_names["qwen"], "hy_mt": model_names["hy_mt"], "translategemma": model_names["translategemma"]},
            "gpu": gpu,
            "device": device,
            "max_new_tokens": max_new_tokens_translation,
        },
        timeout=timeout,
    )
    translated = base_df.merge(pd.read_csv(translation_csv), on=ROW_ID_COL, how="left")
    translated_csv = tmp / "base_plus_translations.csv"
    translated.to_csv(translated_csv, index=False)

    green_csv = tmp / "russian_green.csv"
    if verbose:
        print("[build] Russian GREEN/Qwen in isolated process", flush=True)
    run_module_function_in_process(
        "calculate_russian_text",
        "calculate_green_for_csv",
        {
            "input_csv": base_csv,
            "output_csv": str(green_csv),
            "candidate_col": CANDIDATE_COL,
            "reference_col": REFERENCE_COL,
            "row_id_col": ROW_ID_COL,
            "model_name": model_names["green"],
            "batch_size": green_batch_size,
            "gpu": gpu,
            "device": device,
            "max_new_tokens": max_new_tokens_green,
        },
        timeout=timeout,
    )

    text_csv = tmp / "english_text_metrics.csv"
    if verbose:
        print("[build] English text metrics in isolated process", flush=True)
    run_module_function_in_process(
        "calculate_english_text",
        "calculate_english_text_metrics_for_csv",
        {
            "input_csv": str(translated_csv),
            "output_csv": str(text_csv),
            "row_id_col": ROW_ID_COL,
            "translators": list(translators),
            "kinds": list(kinds),
            "methods": list(ALL_ENGLISH_TEXT_METHODS),
            "cxrbert_model": model_names["cxrbert"],
            "biovilt_model": model_names["biovilt"],
            "batch_size": text_batch_size,
            "gpu": gpu,
            "device": device,
        },
        timeout=timeout,
    )

    image_csv = tmp / "image_text_metrics.csv"
    if verbose:
        print("[build] BioViL-T image-text metric in isolated process", flush=True)
    run_module_function_in_process(
        "calculate_image_text",
        "calculate_image_text_metrics_for_csv",
        {
            "input_csv": str(translated_csv),
            "output_csv": str(image_csv),
            "image_col": IMAGE_COL,
            "row_id_col": ROW_ID_COL,
            "translators": list(translators),
            "kinds": list(kinds),
            "image_root": image_root,
            "biovilt_model": model_names["biovilt"],
            "batch_size": image_batch_size,
            "gpu": gpu,
            "device": device,
        },
        timeout=timeout,
    )

    feature_df = translated[[ROW_ID_COL, TARGET_COL]].copy()
    for path in [green_csv, text_csv, image_csv]:
        feature_df = merge_on_row_id(feature_df, path, row_id_col=ROW_ID_COL)
    return feature_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full radiology report quality method: metrics, nested CV, final config.")
    p.add_argument("path_to_data_description", help="CSV with candidate report, reference report, image path and physician score.")
    p.add_argument("--config-out", default="radiology_quality_config.json", help="Path to save selected final ensemble config.")
    p.add_argument("--candidate-col", required=True)
    p.add_argument("--reference-col", required=True)
    p.add_argument("--image-col", required=True)
    p.add_argument("--target-col", required=True)
    p.add_argument("--row-id-col", default=None)
    p.add_argument("--target-direction", choices=["higher_is_better", "lower_is_better"], default="higher_is_better")
    p.add_argument("--translators", default="qwen,hy_mt,translategemma")
    p.add_argument("--translation-kinds", default="terms,noterms")
    p.add_argument("--k-outer", type=int, default=5)
    p.add_argument("--k-inner", type=int, default=5)
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--gpu", default="0")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--timeout", type=int, default=None, help="Per-worker timeout in seconds. Default: no timeout.")
    p.add_argument("--image-root", default=None, help="Root for relative image paths. Default: directory of input CSV.")
    p.add_argument("--green-model", default=DEFAULT_GREEN_MODEL)
    p.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    p.add_argument("--hy-mt-model", default=DEFAULT_HY_MT_MODEL)
    p.add_argument("--translategemma-model", default=DEFAULT_TRANSLATEGEMMA_MODEL)
    p.add_argument("--cxrbert-model", default=DEFAULT_CXRBERT_MODEL)
    p.add_argument("--biovilt-model", default=DEFAULT_BIOVILT_MODEL)
    p.add_argument("--green-batch-size", type=int, default=1)
    p.add_argument("--text-batch-size", type=int, default=16)
    p.add_argument("--image-batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens-translation", type=int, default=256)
    p.add_argument("--max-new-tokens-green", type=int, default=768)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.path_to_data_description).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    config_out = Path(args.config_out).expanduser().resolve()
    config_out.parent.mkdir(parents=True, exist_ok=True)
    image_root = args.image_root or str(input_csv.parent)
    translators = split_csv_arg(args.translators, ["qwen", "hy_mt", "translategemma"])
    kinds = split_csv_arg(args.translation_kinds, ["terms", "noterms"])
    model_names = {
        "green": args.green_model,
        "qwen": args.qwen_model,
        "hy_mt": args.hy_mt_model,
        "translategemma": args.translategemma_model,
        "cxrbert": args.cxrbert_model,
        "biovilt": args.biovilt_model,
    }

    # Make local modules importable inside spawned workers.
    os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")

    with tempfile.TemporaryDirectory(prefix="radiology_quality_build_") as td:
        base_csv = str(Path(td) / "base.csv")
        prep = prepare_base_csv(
            str(input_csv),
            base_csv,
            args.candidate_col,
            args.reference_col,
            args.image_col,
            args.target_col,
            args.row_id_col,
        )
        if args.verbose:
            print(f"[build] prepared {prep['n_rows']} rows", flush=True)
        feature_df = compute_all_features(
            base_csv=base_csv,
            tmp_dir=td,
            translators=translators,
            kinds=kinds,
            gpu=args.gpu,
            device=args.device,
            timeout=args.timeout,
            model_names=model_names,
            green_batch_size=args.green_batch_size,
            text_batch_size=args.text_batch_size,
            image_batch_size=args.image_batch_size,
            image_root=image_root,
            max_new_tokens_translation=args.max_new_tokens_translation,
            max_new_tokens_green=args.max_new_tokens_green,
            verbose=args.verbose,
        )

        selected = select_and_fit_final_config(
            feature_df=feature_df,
            target_col=TARGET_COL,
            translators=translators,
            kinds=kinds,
            k_outer=args.k_outer,
            k_inner=args.k_inner,
            weight_step=args.weight_step,
            random_state=args.random_state,
            target_higher_is_better=args.target_direction == "higher_is_better",
        )

    config: Dict[str, Any] = {
        "schema_version": 1,
        "method_name": "radiology_report_quality_ensemble",
        "input_mapping": prep["mapping"],
        "canonical_columns": {
            "row_id": ROW_ID_COL,
            "candidate": CANDIDATE_COL,
            "reference": REFERENCE_COL,
            "image": IMAGE_COL,
            "target": TARGET_COL,
        },
        "translators": translators,
        "translation_kinds": kinds,
        "model_names": model_names,
        "basic_methods": BASIC_METHODS,
        "image_root": image_root,
        **selected,
    }
    write_json(config_out, config)
    print(f"Saved config: {config_out}")
    print("Selected features:")
    for spec, weight in zip(config["selected_features"], config["weights"]):
        print(f"  {weight:.4f} * {spec['feature_col']} ({spec['method_key']})")
    print(f"Nested outer mean tau: {config['training']['outer_test_mean_tau']:.4f}")


if __name__ == "__main__":
    main()
