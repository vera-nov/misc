from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import utils as u


def merge_on_row_id(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    u.require_column(left, u.ROW_ID_COL)
    u.require_column(right, u.ROW_ID_COL)
    keep = [c for c in right.columns if c != u.ROW_ID_COL and c not in left.columns]
    return left.merge(right[[u.ROW_ID_COL] + keep], on=u.ROW_ID_COL, how="left")


def prepare_base_csv(input_csv: str, tmp_dir: Path, args: argparse.Namespace) -> Tuple[Path, pd.DataFrame, Dict[str, Any]]:
    original = pd.read_csv(input_csv)
    normalized, column_mapping = u.normalize_input_dataframe(
        original,
        candidate_col=args.candidate_col,
        reference_col=args.reference_col,
        image_col=args.image_col,
        target_col=args.target_col,
        require_target=True,
    )
    base_csv = tmp_dir / "base.csv"
    normalized[[u.ROW_ID_COL, u.CANDIDATE_COL, u.REFERENCE_COL, u.IMAGE_COL, u.TARGET_COL]].to_csv(base_csv, index=False)
    return base_csv, normalized, column_mapping


def compute_all_features(args: argparse.Namespace, base_csv: Path, tmp_dir: Path) -> pd.DataFrame:
    from calculate_english_text import calculate_english_text_metrics_for_csv
    from calculate_image_text import calculate_image_text_metrics_for_csv
    from calculate_russian_text import calculate_green_for_csv
    from translate_reports import translate_reports_for_csv

    translated_csv = tmp_dir / "translations.csv"
    green_csv = tmp_dir / "green.csv"
    english_csv = tmp_dir / "english.csv"
    image_csv = tmp_dir / "image.csv"
    merged_for_english = tmp_dir / "base_with_translations.csv"

    translators = u.split_csv_arg(args.translators, u.TRANSLATORS_DEFAULT)
    kinds = u.split_csv_arg(args.translation_kinds, u.TRANSLATION_KINDS_DEFAULT)

    translated = translate_reports_for_csv(
        input_csv=str(base_csv),
        output_csv=str(translated_csv),
        translators=translators,
        translation_kinds=kinds,
        qwen_model=args.qwen_translator_model,
        hy_mt_model=args.hy_mt_model,
        translategemma_model=args.translategemma_model,
        max_new_tokens=args.translation_max_new_tokens,
        device=args.device,
        gpu=args.gpu,
        per_translator_timeout=args.per_translator_timeout,
        verbose=args.verbose,
    )

    base = pd.read_csv(base_csv)
    base_with_translations = merge_on_row_id(base, translated)
    base_with_translations.to_csv(merged_for_english, index=False)

    green = calculate_green_for_csv(
        input_csv=str(base_csv),
        output_csv=str(green_csv),
        model_name=args.green_model,
        batch_size=args.green_batch_size,
        max_new_tokens=args.green_max_new_tokens,
        device=args.device,
        gpu=args.gpu,
        verbose=args.verbose,
    )

    english = calculate_english_text_metrics_for_csv(
        input_csv=str(merged_for_english),
        output_csv=str(english_csv),
        translators=translators,
        translation_kinds=kinds,
        methods=None,
        cxrbert_model=args.cxrbert_model,
        biovilt_model=args.biovilt_model,
        batch_size=args.text_batch_size,
        device=args.device,
        gpu=args.gpu,
        verbose=args.verbose,
    )

    image = calculate_image_text_metrics_for_csv(
        input_csv=str(merged_for_english),
        output_csv=str(image_csv),
        translators=translators,
        translation_kinds=kinds,
        image_root=args.image_root,
        biovilt_model=args.biovilt_model,
        batch_size=args.image_batch_size,
        device=args.device,
        gpu=args.gpu,
        verbose=args.verbose,
    )

    features = base[[u.ROW_ID_COL, u.TARGET_COL]].copy()
    for chunk in (green, english, image):
        features = merge_on_row_id(features, chunk)
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-data-description", required=True)
    parser.add_argument("--candidate-col", required=True)
    parser.add_argument("--reference-col", required=True)
    parser.add_argument("--image-col", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--output-json", default="quality_method_config.json")
    parser.add_argument("--target-direction", choices=["higher", "lower"], default="higher")
    parser.add_argument("--translators", default="qwen,hy_mt,translategemma")
    parser.add_argument("--translation-kinds", default="terms,noterms")
    parser.add_argument("--k-outer", type=int, default=5)
    parser.add_argument("--k-inner", type=int, default=4)
    parser.add_argument("--weight-step", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--per-translator-timeout", type=int, default=3600)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--qwen-translator-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hy-mt-model", default="tencent/HY-MT1.5-7B")
    parser.add_argument("--translategemma-model", default="google/translategemma-12b-it")
    parser.add_argument("--green-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cxrbert-model", default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--biovilt-model", default="microsoft/BiomedVLP-BioViL-T")
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

    translators = u.split_csv_arg(args.translators, u.TRANSLATORS_DEFAULT)
    kinds = u.split_csv_arg(args.translation_kinds, u.TRANSLATION_KINDS_DEFAULT)

    with tempfile.TemporaryDirectory(prefix="metric_build_") as d:
        tmp_dir = Path(d)
        base_csv, normalized, column_mapping = prepare_base_csv(args.path_to_data_description, tmp_dir, args)
        features = compute_all_features(args, base_csv, tmp_dir)
        final = u.select_and_fit_final_config(
            feature_df=features,
            target_col=u.TARGET_COL,
            translators=translators,
            kinds=kinds,
            k_outer=args.k_outer,
            k_inner=args.k_inner,
            weight_step=args.weight_step,
            random_state=args.random_state,
            target_higher_is_better=args.target_direction == "higher",
        )

    payload: Dict[str, Any] = {
        "version": 1,
        "columns": {
            "row_id": u.ROW_ID_COL,
            "candidate_report": u.CANDIDATE_COL,
            "reference_report": u.REFERENCE_COL,
            "image_path": u.IMAGE_COL,
            "human_score": u.TARGET_COL,
            "candidate_col_source": column_mapping["candidate_col_source"],
            "reference_col_source": column_mapping["reference_col_source"],
            "image_col_source": column_mapping["image_col_source"],
            "target_col_source": column_mapping["target_col_source"],
            "row_id_col_source": column_mapping["row_id_col_source"],
        },
        "target_direction": args.target_direction,
        "selected_features": final["selected_features"],
        "weights": final["weights"],
        "standardizer": final["standardizer"],
        "standardization": final["standardizer"],
        "training": final["training"],
        "nested_cv_outer_results": final["nested_cv_outer_results"],
        "cv": final["training"],
        "n_train_rows": int(len(normalized)),
        "models": {
            "qwen_translator_model": args.qwen_translator_model,
            "hy_mt_model": args.hy_mt_model,
            "translategemma_model": args.translategemma_model,
            "green_model": args.green_model,
            "cxrbert_model": args.cxrbert_model,
            "biovilt_model": args.biovilt_model,
        },
        "runtime": {
            "translators": translators,
            "translation_kinds": kinds,
            "translation_max_new_tokens": args.translation_max_new_tokens,
            "green_batch_size": args.green_batch_size,
            "green_max_new_tokens": args.green_max_new_tokens,
            "text_batch_size": args.text_batch_size,
            "image_batch_size": args.image_batch_size,
        },
    }

    u.write_json(args.output_json, payload)
    print(u.json_dumps({"output_json": args.output_json, "cv": final["training"], "selected_features": final["selected_features"], "weights": final["weights"]}))


if __name__ == "__main__":
    main()
