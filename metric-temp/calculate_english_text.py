from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import ROW_ID_COL, cleanup_cuda, configure_worker_environment, get_input_device, require_column

DEFAULT_CXRBERT_MODEL = "microsoft/BiomedVLP-CXR-BERT-specialized"
DEFAULT_BIOVILT_MODEL = "microsoft/BiomedVLP-BioViL-T"
ALL_ENGLISH_TEXT_METHODS = [
    "radgraph_partial",
    "radcliq",
    "ratescore",
    "cosinesim_cxrbert",
    "cosinesim_biovilt",
    "bertscore_cxrbert",
    "bertscore_biovilt",
]


def safe_texts(values: Sequence[Any]) -> List[str]:
    return ["" if pd.isna(v) else str(v) for v in values]


def pair_columns(df: pd.DataFrame, translators: Sequence[str], kinds: Sequence[str]) -> List[Tuple[str, str, str, str]]:
    pairs = []
    for translator in translators:
        for kind in kinds:
            cand = f"generation_{translator}_{kind}"
            ref = f"gt_{translator}_{kind}"
            require_column(df, cand)
            require_column(df, ref)
            pairs.append((translator, kind, cand, ref))
    return pairs


def calculate_radeval(df: pd.DataFrame, pairs: Sequence[Tuple[str, str, str, str]], methods: Sequence[str]) -> pd.DataFrame:
    from RadEval import RadEval

    needed = [m for m in methods if m in {"radgraph_partial", "radcliq", "ratescore"}]
    out = pd.DataFrame({ROW_ID_COL: df[ROW_ID_COL]})
    if not needed:
        return out
    evaluator = RadEval(
        do_radgraph=True,
        do_green=False,
        do_ratescore=True,
        do_radcliq=True,
        do_srrbert=False,
        do_crimson=False,
        do_per_sample=True,
        show_progress=False,
    )
    mapping = {
        "radgraph_partial": ("radgraph_partial", "text_en_radgraph"),
        "radcliq": ("radcliq_v1", "text_en_radcliq"),
        "ratescore": ("ratescore", "text_en_ratescore"),
    }
    for translator, kind, cand_col, ref_col in pairs:
        result = evaluator(refs=safe_texts(df[ref_col]), hyps=safe_texts(df[cand_col]))
        for method in needed:
            source_key, prefix = mapping[method]
            out[f"{prefix}_{translator}_{kind}"] = result[source_key]
    cleanup_cuda(evaluator)
    return out


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token


def hidden(output: Any):
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    raise ValueError("Model output has no hidden states")


def special_token_mask(input_ids: Any, attention_mask: Any, tokenizer: Any):
    import torch

    mask = attention_mask.bool()
    for token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
        if token_id is not None:
            mask &= input_ids.ne(token_id)
    return mask


def load_text_encoder(model_name: str, device: Optional[str], biovilt: bool = False):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    actual_device = get_input_device(device)
    if biovilt:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.tie_word_embeddings = False
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False, device_map=None)
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=False, device_map=None)
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    actual_device = "cuda" if actual_device != "cpu" and torch.cuda.is_available() else "cpu"
    model.to(actual_device)
    model.eval()
    return tokenizer, model


def text_embeddings(texts: Sequence[str], tokenizer: Any, model: Any, biovilt: bool, batch_size: int = 16, max_length: int = 512) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        batch = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        if biovilt:
            batch.pop("token_type_ids", None)

        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.inference_mode():
            if biovilt and hasattr(model, "get_projected_text_embeddings"):
                emb = model.get_projected_text_embeddings(**batch)
            else:
                emb = hidden(model(**batch, return_dict=True))[:, 0, :]

        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.detach().cpu().numpy())

    cleanup_cuda()
    return np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 1), dtype=np.float32)


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


def bertscore_f1(cands: Sequence[str], refs: Sequence[str], tokenizer: Any, model: Any, batch_size: int = 8, max_length: int = 512) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    scores: List[float] = []
    for start in range(0, len(cands), batch_size):
        cand_batch = tokenizer(list(cands[start : start + batch_size]), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        ref_batch = tokenizer(list(refs[start : start + batch_size]), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        cand_batch.pop("token_type_ids", None)
        ref_batch.pop("token_type_ids", None)

        cand_batch = {k: v.to(model.device) for k, v in cand_batch.items()}
        ref_batch = {k: v.to(model.device) for k, v in ref_batch.items()}
        with torch.inference_mode():
            cand_h = F.normalize(hidden(model(**cand_batch, return_dict=True)), dim=-1)
            ref_h = F.normalize(hidden(model(**ref_batch, return_dict=True)), dim=-1)
        cand_mask = special_token_mask(cand_batch["input_ids"], cand_batch["attention_mask"], tokenizer)
        ref_mask = special_token_mask(ref_batch["input_ids"], ref_batch["attention_mask"], tokenizer)
        for i in range(cand_h.shape[0]):
            c = cand_h[i][cand_mask[i]]
            r = ref_h[i][ref_mask[i]]
            if c.numel() == 0 or r.numel() == 0:
                scores.append(0.0)
                continue
            sim = c @ r.T
            p = sim.max(dim=1).values.mean()
            rr = sim.max(dim=0).values.mean()
            f1 = 2 * p * rr / (p + rr + 1e-12)
            scores.append(float(f1.detach().cpu()))
    cleanup_cuda()
    return np.asarray(scores, dtype=np.float32)


def calculate_encoder_metrics(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str, str, str]],
    methods: Sequence[str],
    model_key: str,
    model_name: str,
    device: Optional[str],
    batch_size: int,
) -> pd.DataFrame:
    need_cosine = f"cosinesim_{model_key}" in methods
    need_bert = f"bertscore_{model_key}" in methods
    out = pd.DataFrame({ROW_ID_COL: df[ROW_ID_COL]})
    if not need_cosine and not need_bert:
        return out
    biovilt = model_key == "biovilt"
    tokenizer, model = load_text_encoder(model_name, device, biovilt=biovilt)
    for translator, kind, cand_col, ref_col in pairs:
        cands = safe_texts(df[cand_col])
        refs = safe_texts(df[ref_col])
        if need_cosine:
            cand_emb = text_embeddings(cands, tokenizer, model, biovilt=biovilt, batch_size=batch_size)
            ref_emb = text_embeddings(refs, tokenizer, model, biovilt=biovilt, batch_size=batch_size)
            out[f"text_en_cosinesim_{model_key}_{translator}_{kind}"] = cosine_scores(cand_emb, ref_emb)
        if need_bert:
            out[f"text_en_bertscore_{model_key}_{translator}_{kind}"] = bertscore_f1(cands, refs, tokenizer, model, batch_size=max(1, min(batch_size, 8)))
    cleanup_cuda(model, tokenizer)
    return out


def calculate_english_text_metrics_for_csv(
    input_csv: str,
    output_csv: str,
    translators: Optional[Sequence[str]] = None,
    translation_kinds: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    cxrbert_model: str = DEFAULT_CXRBERT_MODEL,
    biovilt_model: str = DEFAULT_BIOVILT_MODEL,
    batch_size: int = 16,
    device: Optional[str] = None,
    gpu: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    configure_worker_environment(gpu)
    translators = list(translators or ["qwen", "hy_mt", "translategemma"])
    translation_kinds = list(translation_kinds or ["terms", "noterms"])
    methods = list(methods or ALL_ENGLISH_TEXT_METHODS)
    df = pd.read_csv(input_csv)
    require_column(df, ROW_ID_COL)
    pairs = pair_columns(df, translators, translation_kinds)
    out = pd.DataFrame({ROW_ID_COL: df[ROW_ID_COL]})
    chunks = [
        calculate_radeval(df, pairs, methods),
        calculate_encoder_metrics(df, pairs, methods, "cxrbert", cxrbert_model, device, batch_size),
        calculate_encoder_metrics(df, pairs, methods, "biovilt", biovilt_model, device, batch_size),
    ]
    for chunk in chunks:
        for col in chunk.columns:
            if col != ROW_ID_COL:
                out[col] = chunk[col]
    out.to_csv(output_csv, index=False)
    cleanup_cuda()
    return out
