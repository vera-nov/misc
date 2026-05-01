from __future__ import annotations

import gc
import importlib
import json
import math
import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import KFold

ROW_ID_COL = "report_id"
CANDIDATE_COL = "candidate_report"
REFERENCE_COL = "reference_report"
IMAGE_COL = "image_path"
TARGET_COL = "human_score"
EPS = 1e-12

TRANSLATORS_DEFAULT = ["qwen", "hy_mt", "translategemma"]
TRANSLATION_KINDS_DEFAULT = ["terms", "noterms"]

BASIC_METHODS: List[Dict[str, Any]] = [
    {
        "key": "ru_GREEN_Qwen",
        "name": "Qwen GREEN Russian",
        "group": "ru_text",
        "can_process_data_type": {"text": True, "image": False},
        "is_explainable": True,
        "supports_russian": True,
        "requires_translation": False,
        "supports_image": False,
        "higher_is_better": True,
        "feature_col": "ru_GREEN_Qwen",
        "errors_col": "ru_GREEN_Qwen_errors_json",
    },
    {
        "key": "radgraph_partial",
        "name": "RadEval RadGraph partial",
        "group": "english_text",
        "can_process_data_type": {"text": True, "image": False},
        "is_explainable": False,
        "supports_russian": False,
        "requires_translation": True,
        "supports_image": False,
        "higher_is_better": True,
        "feature_template": "text_en_radgraph_{translator}_{kind}",
    },
    {
        "key": "radcliq",
        "name": "RadEval RadCliQ v1",
        "group": "english_text",
        "can_process_data_type": {"text": True, "image": False},
        "is_explainable": False,
        "supports_russian": False,
        "requires_translation": True,
        "supports_image": False,
        "higher_is_better": False,
        "feature_template": "text_en_radcliq_{translator}_{kind}",
    },
    {
        "key": "ratescore",
        "name": "RadEval RateScore",
        "group": "english_text",
        "can_process_data_type": {"text": True, "image": False},
        "is_explainable": False,
        "supports_russian": False,
        "requires_translation": True,
        "supports_image": False,
        "higher_is_better": True,
        "feature_template": "text_en_ratescore_{translator}_{kind}",
    },
    {
        "key": "cosinesim_cxrbert",
        "name": "CXR-BERT-specialized cosine",
        "group": "english_text",
        "can_process_data_type": {"text": True, "image": False},
        "is_explainable": False,
        "supports_russian": False,
        "requires_translation": True,
        "supports_image": False,
        "higher_is_better": True,
        "feature_template": "text_en_cosinesim_cxrbert_{translator}_{kind}",
    },
    {
        "key": "cosinesim_biovilt",
        "name": "BioViL-T text cosine",
        "group": "english_text",
        "can_process_data_type": {"text": True, "image": False},
        "is_explainable": False,
        "supports_russian": False,
        "requires_translation": True,
        "supports_image": False,
        "higher_is_better": True,
        "feature_template": "text_en_cosinesim_biovilt_{translator}_{kind}",
    },
    {
        "key": "bertscore_cxrbert",
        "name": "CXR-BERT-specialized BERTScore-like F1",
        "group": "english_text",
        "can_process_data_type": {"text": True, "image": False},
        "is_explainable": False,
        "supports_russian": False,
        "requires_translation": True,
        "supports_image": False,
        "higher_is_better": True,
        "feature_template": "text_en_bertscore_cxrbert_{translator}_{kind}",
    },
    {
        "key": "bertscore_biovilt",
        "name": "BioViL-T BERTScore-like F1",
        "group": "english_text",
        "can_process_data_type": {"text": True, "image": False},
        "is_explainable": False,
        "supports_russian": False,
        "requires_translation": True,
        "supports_image": False,
        "higher_is_better": True,
        "feature_template": "text_en_bertscore_biovilt_{translator}_{kind}",
    },
    {
        "key": "img_biovilt",
        "name": "BioViL-T image-text cosine",
        "group": "image_text",
        "can_process_data_type": {"text": True, "image": True},
        "is_explainable": False,
        "supports_russian": False,
        "requires_translation": True,
        "supports_image": True,
        "higher_is_better": True,
        "feature_template": "img_en_biovilt_{translator}_{kind}",
    },
]

METHOD_BY_KEY = {m["key"]: m for m in BASIC_METHODS}

@dataclass(frozen=True)
class FeatureSpec:
    method_key: str
    feature_col: str
    group: str
    higher_is_better: bool
    translator: Optional[str] = None
    kind: Optional[str] = None
    is_explainable: bool = False
    errors_col: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method_key": self.method_key,
            "feature_col": self.feature_col,
            "group": self.group,
            "higher_is_better": self.higher_is_better,
            "translator": self.translator,
            "kind": self.kind,
            "is_explainable": self.is_explainable,
            "errors_col": self.errors_col,
        }

@dataclass(frozen=True)
class CandidateConfig:
    ru_feature: FeatureSpec
    text_feature: FeatureSpec
    image_feature: FeatureSpec

    @property
    def features(self) -> List[FeatureSpec]:
        return [self.ru_feature, self.text_feature, self.image_feature]

    @property
    def key(self) -> str:
        return " + ".join(f.feature_col for f in self.features)

    def as_dict(self) -> Dict[str, Any]:
        return {"features": [f.as_dict() for f in self.features], "key": self.key}


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json_dumps(obj), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_csv_arg(x: Optional[str], default: Optional[Sequence[str]] = None) -> List[str]:
    if x is None or str(x).strip() == "":
        return list(default or [])
    return [v.strip() for v in str(x).split(",") if v.strip()]


def require_column(df: pd.DataFrame, col: Optional[str], arg_name: Optional[str] = None) -> str:
    if col is None or str(col).strip() == "":
        name = arg_name or "column"
        raise ValueError(f"Argument '{name}' is required. Available columns: {list(df.columns)}")
    col = str(col).strip()
    if col not in df.columns:
        if arg_name:
            raise ValueError(f"Column passed via '{arg_name}' was not found: '{col}'. Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required column: {col}. Available columns: {list(df.columns)}")
    return col


def normalize_input_dataframe(
    df: pd.DataFrame,
    candidate_col: str,
    reference_col: str,
    image_col: str,
    target_col: Optional[str] = None,
    require_target: bool = True,
    row_id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    work = df.copy()

    c_col = require_column(work, candidate_col, "candidate_col")
    r_col = require_column(work, reference_col, "reference_col")
    i_col = require_column(work, image_col, "image_col")

    if require_target:
        t_col = require_column(work, target_col, "target_col")
    else:
        if target_col is not None and str(target_col).strip() != "":
            t_col = require_column(work, target_col, "target_col")
        else:
            t_col = None

    if row_id_col is not None and str(row_id_col).strip() != "":
        id_col = require_column(work, row_id_col, "row_id_col")
    else:
        id_col = ROW_ID_COL
        if id_col not in work.columns:
            work[id_col] = [str(i) for i in range(len(work))]

    work[CANDIDATE_COL] = work[c_col]
    work[REFERENCE_COL] = work[r_col]
    work[IMAGE_COL] = work[i_col]
    work[ROW_ID_COL] = work[id_col]

    if t_col is not None:
        work[TARGET_COL] = work[t_col]

    work[CANDIDATE_COL] = work[CANDIDATE_COL].fillna("").astype(str)
    work[REFERENCE_COL] = work[REFERENCE_COL].fillna("").astype(str)
    work[IMAGE_COL] = work[IMAGE_COL].fillna("").astype(str)
    work[ROW_ID_COL] = work[ROW_ID_COL].astype(str)

    if require_target:
        work[TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce")
        if work[TARGET_COL].isna().any():
            bad_n = int(work[TARGET_COL].isna().sum())
            raise ValueError(
                f"Target column '{t_col}' contains {bad_n} non-numeric or missing values."
            )

    mapping = {
        "candidate_col_source": c_col,
        "reference_col_source": r_col,
        "image_col_source": i_col,
        "target_col_source": t_col,
        "row_id_col_source": id_col,
    }

    return work, mapping


def pick_dtype(torch_module: Any = None) -> Any:
    if torch_module is None:
        import torch as torch_module
    if torch_module.cuda.is_available():
        try:
            if torch_module.cuda.is_bf16_supported():
                return torch_module.bfloat16
        except Exception:
            pass
        return torch_module.float16
    return torch_module.float32


def get_input_device(device: Optional[str] = None) -> str:
    if device:
        return str(device)
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def cleanup_cuda(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def configure_worker_environment(gpu: Optional[int] = None, device: Optional[str] = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if gpu is not None and (device is None or str(device).startswith("cuda") or str(device) == "auto"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def _process_entry(module_name: str, function_name: str, kwargs: Dict[str, Any], queue: Any) -> None:
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)
        value = fn(**kwargs)
        cleanup_cuda()
        queue.put({"ok": True, "value": value})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def run_module_function_in_process(
    module_name: str,
    function_name: str,
    kwargs: Dict[str, Any],
    timeout: Optional[int] = None,
) -> Any:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_process_entry, args=(module_name, function_name, kwargs, queue))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=20)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=20)
        raise TimeoutError(f"Worker {module_name}.{function_name} exceeded timeout={timeout}s")
    if queue.empty():
        raise RuntimeError(f"Worker {module_name}.{function_name} exited with code {proc.exitcode} and returned no result")
    payload = queue.get()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("traceback") or payload.get("error"))
    return payload.get("value")


def safe_kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if len(xa) < 2 or np.nanstd(xa) < EPS or np.nanstd(ya) < EPS:
        return 0.0
    tau, _ = kendalltau(xa, ya)
    if tau is None or np.isnan(tau):
        return 0.0
    return float(tau)


def build_weight_grid(
    n_features: int,
    step: float,
    required_nonzero_indices: Optional[Sequence[int]] = [0, 2],
) -> np.ndarray:
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    if step <= 0 or step > 1:
        raise ValueError("weight_step must be in (0, 1].")

    inv = round(1.0 / step)
    if not math.isclose(inv * step, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("weight_step must evenly divide 1.0, e.g. 0.1, 0.05, 0.01")

    required_nonzero_indices = list(required_nonzero_indices or [])

    for idx in required_nonzero_indices:
        if idx < 0 or idx >= n_features:
            raise ValueError(f"required_nonzero index {idx} is out of range for {n_features} features")
    min_units = [0] * n_features
    for idx in required_nonzero_indices:
        min_units[idx] = 1

    min_sum = sum(min_units)
    if min_sum > inv:
        raise ValueError(
            f"Impossible constraints: {len(required_nonzero_indices)} required nonzero weights "
            f"with step={step} need at least {min_sum * step}, but total weight is 1.0"
        )

    rows: List[List[float]] = []
    remaining_total = inv - min_sum

    def rec(prefix: List[int], remaining: int, k_left: int) -> None:
        if k_left == 1:
            units = prefix + [remaining]
            full_units = [u + m for u, m in zip(units, min_units)]
            rows.append([v * step for v in full_units])
            return

        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v, k_left - 1)

    rec([], remaining_total, n_features)
    return np.asarray(rows, dtype=float)


def feature_specs_from_registry(translators: Sequence[str], kinds: Sequence[str]) -> Tuple[List[FeatureSpec], List[FeatureSpec], List[FeatureSpec]]:
    """
    create FeatureSpec objects from BASIC_METHODS
    """
    ru: List[FeatureSpec] = []
    text: List[FeatureSpec] = []
    image: List[FeatureSpec] = []
    for m in BASIC_METHODS:
        if m["group"] == "ru_text":
            ru.append(FeatureSpec(
                method_key=m["key"],
                feature_col=m["feature_col"],
                group=m["group"],
                higher_is_better=bool(m["higher_is_better"]),
                is_explainable=bool(m.get("is_explainable", False)),
                errors_col=m.get("errors_col"),
            ))
        elif m["group"] in {"english_text", "image_text"}:
            for tr in translators:
                for kind in kinds:
                    if tr == "translategemma" and kind == "terms":
                        # TranslateGemma does not accept glossary injection
                        continue
                    spec = FeatureSpec(
                        method_key=m["key"],
                        feature_col=m["feature_template"].format(translator=tr, kind=kind),
                        group=m["group"],
                        higher_is_better=bool(m["higher_is_better"]),
                        translator=tr,
                        kind=kind,
                        is_explainable=bool(m.get("is_explainable", False)),
                    )
                    (text if m["group"] == "english_text" else image).append(spec)
    return ru, text, image


def make_candidate_configs(
        available_cols: Iterable[str],
        translators: Sequence[str],
        kinds: Sequence[str]) -> List[CandidateConfig]:
    """
    create all available basic method configurations
    """
    cols = set(available_cols)
    ru_specs, text_specs, image_specs = feature_specs_from_registry(translators, kinds)
    ru_specs = [s for s in ru_specs if s.feature_col in cols]
    text_specs = [s for s in text_specs if s.feature_col in cols]
    image_specs = [s for s in image_specs if s.feature_col in cols]
    configs: List[CandidateConfig] = []
    for ru in ru_specs:
        for text in text_specs:
            for image in image_specs:
                if text.translator == image.translator and text.kind == image.kind:
                    configs.append(CandidateConfig(ru, text, image))
    if not configs:
        missing_msg = {
            "ru_features_found": [s.feature_col for s in ru_specs],
            "text_features_found": [s.feature_col for s in text_specs][:10],
            "image_features_found": [s.feature_col for s in image_specs][:10],
        }
        raise ValueError(f"No complete candidate configs could be built. Details: {missing_msg}")
    return configs


def _quality_feature_matrix(df: pd.DataFrame, specs: Sequence[FeatureSpec]) -> np.ndarray:
    cols = [s.feature_col for s in specs]
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    result = x.copy()
    for j, spec in enumerate(specs):
        if not spec.higher_is_better:
            col = x[:, j]
            result[:, j] = np.where(col == 0, 0.0, 1.0 / col)
    return result


def _fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    stds = np.where((~np.isfinite(stds)) | (stds < EPS), 1.0, stds)
    means = np.where(np.isfinite(means), means, 0.0)
    return means, stds


def _transform(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (x - means) / stds


def _optimize_weights(z_train: np.ndarray, y_train: np.ndarray, weight_grid: np.ndarray) -> Tuple[np.ndarray, float]:
    best_w: Optional[np.ndarray] = None
    best_tau = -1e18
    for w in weight_grid:
        scores = z_train @ w
        tau = safe_kendall_tau(scores, y_train)
        if best_w is None or tau > best_tau + 1e-12 or (abs(tau - best_tau) <= 1e-12 and tuple(w.tolist()) < tuple(best_w.tolist())):
            best_w = w.copy()
            best_tau = tau
    assert best_w is not None
    return best_w, float(best_tau)


def _evaluate_config_cv(
    df: pd.DataFrame,
    target_quality: np.ndarray,
    config: CandidateConfig,
    folds: KFold,
    weight_grid: np.ndarray,
) -> Tuple[List[float], List[List[float]]]:
    """
    evaluate one metric config with cross validation
    """
    x = _quality_feature_matrix(df, config.features)
    taus: List[float] = []
    weights: List[List[float]] = []
    for train_idx, val_idx in folds.split(df):
        x_train = x[train_idx]
        x_val = x[val_idx]
        y_train = target_quality[train_idx]
        y_val = target_quality[val_idx]
        means, stds = _fit_standardizer(x_train)
        z_train = _transform(x_train, means, stds)
        z_val = _transform(x_val, means, stds)
        w, _ = _optimize_weights(z_train, y_train, weight_grid)
        val_scores = z_val @ w
        taus.append(safe_kendall_tau(val_scores, y_val))
        weights.append(w.tolist())
    return taus, weights


def select_and_fit_final_config(
    feature_df: pd.DataFrame,
    target_col: str,
    translators: Sequence[str],
    kinds: Sequence[str],
    k_outer: int = 5,
    k_inner: int = 5,
    weight_step: float = 0.05,
    random_state: int = 42,
    target_higher_is_better: bool = True,
) -> Dict[str, Any]:
    candidates = make_candidate_configs(feature_df.columns, translators, kinds)
    needed_cols = sorted({c for cfg in candidates for c in [s.feature_col for s in cfg.features]} | {target_col})
    work = feature_df.copy()
    for c in needed_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    before = len(work)
    work[needed_cols] = work[needed_cols].fillna(0)
    if len(work) < max(2, k_outer, k_inner):
        raise ValueError(f"Not enough complete rows after metric calculation: {len(work)} rows, before={before}")
    target = work[target_col].to_numpy(dtype=float)
    target_quality = target if target_higher_is_better else -target
    weight_grid = build_weight_grid(3, weight_step)

    outer_rows: List[Dict[str, Any]] = []
    outer_cv = KFold(n_splits=k_outer, shuffle=True, random_state=random_state)
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(work), start=1):
        train_df = work.iloc[outer_train_idx].reset_index(drop=True)
        test_df = work.iloc[outer_test_idx].reset_index(drop=True)
        train_target = target_quality[outer_train_idx]
        test_target = target_quality[outer_test_idx]
        inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=random_state + outer_fold)
        scored: List[Tuple[float, float, str, CandidateConfig, List[float]]] = []
        for cfg in candidates:
            taus, _weights = _evaluate_config_cv(train_df, train_target, cfg, inner_cv, weight_grid)
            scored.append((float(np.mean(taus)), float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0, cfg.key, cfg, taus))
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        selected = scored[0][3]
        x_train = _quality_feature_matrix(train_df, selected.features)
        x_test = _quality_feature_matrix(test_df, selected.features)
        means, stds = _fit_standardizer(x_train)
        w, train_tau = _optimize_weights(_transform(x_train, means, stds), train_target, weight_grid)
        test_scores = _transform(x_test, means, stds) @ w
        outer_rows.append({
            "outer_fold": outer_fold,
            "selected_key": selected.key,
            "features": [s.as_dict() for s in selected.features],
            "weights": w.tolist(),
            "inner_mean_tau": scored[0][0],
            "inner_std_tau": scored[0][1],
            "outer_train_tau": train_tau,
            "outer_test_tau": safe_kendall_tau(test_scores, test_target),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        })

    full_cv = KFold(n_splits=k_inner, shuffle=True, random_state=random_state + 10_000)
    final_scored: List[Tuple[float, float, str, CandidateConfig, List[float]]] = []
    for cfg in candidates:
        taus, _weights = _evaluate_config_cv(work, target_quality, cfg, full_cv, weight_grid)
        final_scored.append((float(np.mean(taus)), float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0, cfg.key, cfg, taus))
    final_scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    final_cfg = final_scored[0][3]
    x_all = _quality_feature_matrix(work, final_cfg.features)
    means, stds = _fit_standardizer(x_all)
    z_all = _transform(x_all, means, stds)
    weights, train_tau = _optimize_weights(z_all, target_quality, weight_grid)
    scores = z_all @ weights

    return {
        "selected_config": final_cfg.as_dict(),
        "selected_features": [s.as_dict() for s in final_cfg.features],
        "weights": weights.tolist(),
        "standardizer": {"means": means.tolist(), "stds": stds.tolist()},
        "target_col": target_col,
        "target_higher_is_better": target_higher_is_better,
        "training": {
            "n_rows_input": int(before),
            "n_rows_complete": int(len(work)),
            "weight_step": float(weight_step),
            "n_weight_grid_points": int(len(weight_grid)),
            "k_outer": int(k_outer),
            "k_inner": int(k_inner),
            "random_state": int(random_state),
            "final_cv_mean_tau": final_scored[0][0],
            "final_cv_std_tau": final_scored[0][1],
            "final_train_tau": train_tau,
            "outer_test_mean_tau": float(np.mean([r["outer_test_tau"] for r in outer_rows])),
            "outer_test_std_tau": float(np.std([r["outer_test_tau"] for r in outer_rows], ddof=1)) if len(outer_rows) > 1 else 0.0,
            "score_mean_on_training": float(np.mean(scores)),
            "score_std_on_training": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        },
        "nested_cv_outer_results": outer_rows,
    }


def score_with_config(feature_df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
    specs = [FeatureSpec(**s) for s in config["selected_features"]]
    x = _quality_feature_matrix(feature_df, specs)
    standardizer = config.get("standardizer") or config.get("standardization")
    means = np.asarray(standardizer["means"], dtype=float)
    stds = np.asarray(standardizer["stds"], dtype=float)
    weights = np.asarray(config["weights"], dtype=float)
    return _transform(x, means, stds) @ weights


def selected_translators_and_kinds(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    pairs = []
    for spec in config.get("selected_features", []):
        tr = spec.get("translator")
        kind = spec.get("kind")
        if tr and kind:
            pairs.append((tr, kind))
    trs = sorted({p[0] for p in pairs})
    kinds = sorted({p[1] for p in pairs})
    return trs, kinds


def selected_english_metric_keys(config: Dict[str, Any]) -> List[str]:
    return sorted({s["method_key"] for s in config.get("selected_features", []) if s.get("group") == "english_text"})


def config_needs_green(config: Dict[str, Any]) -> bool:
    return any(s.get("method_key") == "ru_GREEN_Qwen" for s in config.get("selected_features", []))


def config_needs_image(config: Dict[str, Any]) -> bool:
    return any(s.get("group") == "image_text" for s in config.get("selected_features", []))
