from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import ROW_ID_COL, cleanup_cuda, configure_worker_environment, get_input_device, split_csv_arg

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


def safe_texts(xs: Sequence[Any]) -> List[str]:
    return ["" if pd.isna(x) else str(x) for x in xs]


def pair_columns(df: pd.DataFrame, translators: Sequence[str], kinds: Sequence[str]) -> List[Tuple[str, str, str, str]]:
    pairs: List[Tuple[str, str, str, str]] = []
    for tr in translators:
        for kind in kinds:
            ref_col = f"gt_{tr}_{kind}"
            hyp_col = f"generation_{tr}_{kind}"
            if ref_col in df.columns and hyp_col in df.columns:
                pairs.append((tr, kind, ref_col, hyp_col))
    if not pairs:
        raise ValueError("No translation pairs found. Expected columns like gt_qwen_noterms and generation_qwen_noterms.")
    return pairs


def calculate_radeval(df: pd.DataFrame, pairs: Sequence[Tuple[str, str, str, str]], methods: Sequence[str]) -> pd.DataFrame:
    requested = set(methods)
    if not requested.intersection({"radgraph_partial", "radcliq", "ratescore"}):
        return pd.DataFrame(index=df.index)
    from RadEval import RadEval

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
    key_map = {
        "radgraph_partial": ("radgraph_partial", "text_en_radgraph"),
        "radcliq": ("radcliq_v1", "text_en_radcliq"),
        "ratescore": ("ratescore", "text_en_ratescore"),
    }
    out = pd.DataFrame(index=df.index)
    for tr, kind, ref_col, hyp_col in pairs:
        res = evaluator(refs=safe_texts(df[ref_col].tolist()), hyps=safe_texts(df[hyp_col].tolist()))
        for method_key, (radeval_key, prefix) in key_map.items():
            if method_key in requested:
                out[f"{prefix}_{tr}_{kind}"] = np.asarray(res[radeval_key], dtype=float)
    cleanup_cuda(evaluator)
    return out


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def hidden(outputs: Any) -> Any:
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, dict) and "last_hidden_state" in outputs:
        return outputs["last_hidden_state"]
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    raise RuntimeError("last_hidden_state not found in model output")


def special_token_mask(tokenizer: Any, input_ids: Any, attention_mask: Any) -> Any:
    mask = attention_mask.bool().clone()
    for token_id in set(getattr(tokenizer, "all_special_ids", []) or []):
        mask &= input_ids.ne(token_id)
    return mask


def load_text_encoder(model_name: str, device: str, biovilt: bool = False) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    if biovilt:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.tie_word_embeddings = False
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False, device_map=None)
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=False, device_map=None)
    emb = model.get_input_embeddings()
    if emb is not None and len(tokenizer) > getattr(emb, "num_embeddings", len(tokenizer)):
        model.resize_token_embeddings(len(tokenizer))
    actual_device = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    model = model.to(actual_device)
    model.eval()
    return tokenizer, model


def text_embeddings(texts: Sequence[str], tokenizer: Any, model: Any, biovilt: bool, batch_size: int = 16, max_length: int = 512) -> Any:
    import torch
    import torch.nn.functional as F

    device = get_input_device(model)
    xs = []
    texts = safe_texts(texts)
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i : i + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            if biovilt and hasattr(model, "get_projected_text_embeddings"):
                emb = model.get_projected_text_embeddings(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            else:
                emb = hidden(model(**batch, return_dict=True))[:, 0, :]
            xs.append(F.normalize(emb.detach().float(), p=2, dim=1).cpu())
            cleanup_cuda(batch, emb)
    return torch.cat(xs, dim=0)


def cosine_scores(a: Any, b: Any) -> np.ndarray:
    import torch.nn.functional as F

    a = F.normalize(a.float(), p=2, dim=1)
    b = F.normalize(b.float(), p=2, dim=1)
    return (a * b).sum(dim=1).cpu().numpy()


def bertscore_f1(
    cands: Sequence[str],
    refs: Sequence[str],
    tokenizer: Any,
    model: Any,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    cands = safe_texts(cands)
    refs = safe_texts(refs)
    device = get_input_device(model)
    scores: List[float] = []
    with torch.inference_mode():
        for i in range(0, len(cands), batch_size):
            cb = cands[i : i + batch_size]
            rb = refs[i : i + batch_size]
            ci = tokenizer(cb, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            ri = tokenizer(rb, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            ci = {k: v.to(device) for k, v in ci.items()}
            ri = {k: v.to(device) for k, v in ri.items()}
            co = model(**ci, return_dict=True)
            ro = model(**ri, return_dict=True)
            ch = F.normalize(hidden(co).float(), p=2, dim=-1)
            rh = F.normalize(hidden(ro).float(), p=2, dim=-1)
            cm = special_token_mask(tokenizer, ci["input_ids"], ci["attention_mask"])
            rm = special_token_mask(tokenizer, ri["input_ids"], ri["attention_mask"])
            for j in range(ch.size(0)):
                c = ch[j][cm[j]]
                r = rh[j][rm[j]]
                if c.numel() == 0 or r.numel() == 0:
                    f1 = 0.0
                else:
                    sim = c @ r.T
                    p = sim.max(dim=1).values.mean()
                    rr = sim.max(dim=0).values.mean()
                    f1 = float((2 * p * rr / (p + rr + 1e-12)).detach().cpu())
                scores.append(f1)
            cleanup_cuda(ci, ri, co, ro, ch, rh, cm, rm)
    return np.asarray(scores, dtype=np.float32)


def calculate_encoder_metrics(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str, str, str]],
    methods: Sequence[str],
    model_key: str,
    model_name: str,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    need_cosine = f"cosinesim_{model_key}" in methods
    need_bert = f"bertscore_{model_key}" in methods
    if not need_cosine and not need_bert:
        return pd.DataFrame(index=df.index)
    tokenizer, model = load_text_encoder(model_name, device=device, biovilt=(model_key == "biovilt"))
    out = pd.DataFrame(index=df.index)
    try:
        for tr, kind, ref_col, hyp_col in pairs:
            refs = safe_texts(df[ref_col].tolist())
            hyps = safe_texts(df[hyp_col].tolist())
            if need_cosine:
                ref_emb = text_embeddings(refs, tokenizer, model, biovilt=(model_key == "biovilt"), batch_size=batch_size)
                hyp_emb = text_embeddings(hyps, tokenizer, model, biovilt=(model_key == "biovilt"), batch_size=batch_size)
                out[f"text_en_cosinesim_{model_key}_{tr}_{kind}"] = cosine_scores(ref_emb, hyp_emb)
                cleanup_cuda(ref_emb, hyp_emb)
            if need_bert:
                out[f"text_en_bertscore_{model_key}_{tr}_{kind}"] = bertscore_f1(hyps, refs, tokenizer, model, batch_size=min(batch_size, 8))
    finally:
        cleanup_cuda(tokenizer, model)
    return out


def calculate_english_text_metrics_for_csv(
    input_csv: str,
    output_csv: str,
    row_id_col: str = ROW_ID_COL,
    translators: Optional[Sequence[str]] = None,
    kinds: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    cxrbert_model: str = DEFAULT_CXRBERT_MODEL,
    biovilt_model: str = DEFAULT_BIOVILT_MODEL,
    batch_size: int = 16,
    gpu: str = "0",
    device: str = "cuda",
) -> str:
    configure_worker_environment(gpu=gpu, device=device)
    translators = list(translators or ["qwen", "hy_mt", "translategemma"])
    kinds = list(kinds or ["terms", "noterms"])
    methods = list(methods or ALL_ENGLISH_TEXT_METHODS)
    df = pd.read_csv(input_csv)
    pairs = pair_columns(df, translators, kinds)
    out = pd.DataFrame({row_id_col: df[row_id_col].astype(str).tolist() if row_id_col in df.columns else [str(i) for i in range(len(df))]})

    chunks = [
        calculate_radeval(df, pairs, methods),
        calculate_encoder_metrics(df, pairs, methods, "cxrbert", cxrbert_model, device, batch_size),
        calculate_encoder_metrics(df, pairs, methods, "biovilt", biovilt_model, device, batch_size),
    ]
    for chunk in chunks:
        for col in chunk.columns:
            out[col] = chunk[col].to_numpy()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    cleanup_cuda()
    return output_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calculate English text metrics: RadEval, cosine and BERTScore-like metrics.")
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--row-id-col", default=ROW_ID_COL)
    p.add_argument("--translators", default="qwen,hy_mt,translategemma")
    p.add_argument("--kinds", default="terms,noterms")
    p.add_argument("--methods", default=",".join(ALL_ENGLISH_TEXT_METHODS))
    p.add_argument("--cxrbert-model", default=DEFAULT_CXRBERT_MODEL)
    p.add_argument("--biovilt-model", default=DEFAULT_BIOVILT_MODEL)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpu", default="0")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    calculate_english_text_metrics_for_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        row_id_col=args.row_id_col,
        translators=split_csv_arg(args.translators, []),
        kinds=split_csv_arg(args.kinds, []),
        methods=split_csv_arg(args.methods, ALL_ENGLISH_TEXT_METHODS),
        cxrbert_model=args.cxrbert_model,
        biovilt_model=args.biovilt_model,
        batch_size=args.batch_size,
        gpu=args.gpu,
        device=args.device,
    )


if __name__ == "__main__":
    main()
