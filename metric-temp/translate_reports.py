from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import re
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils import (
    CANDIDATE_COL,
    REFERENCE_COL,
    ROW_ID_COL,
    cleanup_cuda,
    configure_worker_environment,
    get_input_device,
    pick_dtype,
    split_csv_arg,
)

warnings.filterwarnings("ignore")

MAX_NEW_TOKENS = 256
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_HY_MT_MODEL = "tencent/HY-MT1.5-7B"
DEFAULT_TRANSLATEGEMMA_MODEL = "google/translategemma-12b-it"

GLOSSARY = [
    ("Apical Cap", "Апикальный козырёк (фиброз)"),
    ("Consolidation", "Консолидация"),
    ("Cyst", "Киста"),
    ("Lobe", "Доля"),
    ("Mass", "Образование"),
    ("Mediastinum", "Отделы средостения"),
    ("Nodule", "Узел"),
    ("Opacity", "Уплотнение"),
    ("Pneumothorax", "Пневмоторакс"),
    ("Pneumonia", "Пневмония"),
    ("Silhouette Sign", "Симптом силуэта"),
    ("Infiltrate", "Инфильтрация"),
]
GLOSSARY_RU_EN = [(ru, en) for en, ru in GLOSSARY]
GLOSSARY_TEXT_RU_EN = "\n".join(f"- {ru} -> {en}" for ru, en in GLOSSARY_RU_EN)


def normalize_source_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.lower() == "nan":
        return ""
    return x


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", str(text), flags=re.DOTALL | re.IGNORECASE)


def maybe_extract_json_translation(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "translation" in obj:
                return str(obj["translation"]).strip()
        except Exception:
            pass
    return text


def clean_translation(text: str) -> str:
    text = normalize_source_text(text)
    text = remove_think_blocks(text)
    text = maybe_extract_json_translation(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip().strip('"').strip("'").strip()


def safe_chat_template(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    for kwargs in (
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {},
    ):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        except TypeError:
            continue
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def load_qwen(model_name: str = DEFAULT_QWEN_MODEL, device: str = "cuda") -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = pick_dtype(torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" and torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "model_id": model_name, "dtype": str(dtype)}


def build_qwen_prompt(text: str, use_terminology: bool = False) -> str:
    base = (
        "You are a medical translator.\n"
        "Translate the following chest X-ray report from Russian to English.\n"
        "Return only the English translation.\n"
        "Do not add explanations, comments, bullet points, or reasoning.\n"
    )
    if use_terminology:
        base += "If one of the following Russian medical terms appears, use the specified English equivalent exactly.\n\n"
        base += f"{GLOSSARY_TEXT_RU_EN}\n\n"
    else:
        base += "\n"
    return base + text


def translate_with_qwen(
    state: Dict[str, Any],
    text: str,
    use_terminology: bool = False,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    import torch

    text = normalize_source_text(text)
    if not text:
        return ""

    tokenizer, model = state["tokenizer"], state["model"]
    messages = [
        {"role": "system", "content": "You are a precise medical translator. Return only the English translation."},
        {"role": "user", "content": build_qwen_prompt(text, use_terminology=use_terminology)},
    ]
    rendered = safe_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(get_input_device(model))
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    out = tokenizer.batch_decode(
        generated[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    cleanup_cuda(inputs, generated)
    return clean_translation(out)


def load_hy_mt(model_name: str = DEFAULT_HY_MT_MODEL, device: str = "cuda") -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = pick_dtype(torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" and torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "model_id": model_name, "dtype": str(dtype)}


def format_glossary_for_hy_mt(glossary_pairs: Sequence[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for tgt, src in glossary_pairs:
        src = str(src).strip()
        tgt = str(tgt).strip()
        if src and tgt:
            lines.append(f"{src} translates as {tgt}")
    return "\n".join(lines)


def build_hy_prompt(text: str, use_terminology: bool = False, target_language: str = "English") -> str:
    if use_terminology:
        glossary_block = format_glossary_for_hy_mt(GLOSSARY)
        if glossary_block:
            return (
                "Refer to the following translations:\n"
                f"{glossary_block}\n\n"
                f"Translate the following segment into {target_language}, without additional explanation.\n\n{text}"
            )
    return f"Translate the following segment into {target_language}, without additional explanation.\n\n{text}"


def clean_hy_translation(text: str) -> str:
    text = clean_translation(text).strip()
    bad_prefixes = [
        "Translate the following segment into English, without additional explanation.",
        "Refer to the following translations:",
        "The translation is:",
        "English translation:",
    ]

    changed = True
    while changed:
        changed = False
        t = text.strip()
        for prefix in bad_prefixes:
            if t.startswith(prefix):
                text = t[len(prefix) :].strip(" \n:,-\"'")
                changed = True
    return text.strip()


def translate_with_hy_mt(
    state: Dict[str, Any],
    text: str,
    use_terminology: bool = False,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    import torch

    text = normalize_source_text(text)
    if not text:
        return ""

    tokenizer, model = state["tokenizer"], state["model"]
    messages = [{"role": "user", "content": build_hy_prompt(text, use_terminology=use_terminology)}]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )

    if isinstance(tokenized, dict):
        model_inputs = {k: v.to(get_input_device(model)) for k, v in tokenized.items()}
        input_ids = model_inputs["input_ids"]
    else:
        input_ids = tokenized.to(get_input_device(model))
        model_inputs = {"input_ids": input_ids}

    input_len = input_ids.shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    out = tokenizer.batch_decode(
        generated[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    cleanup_cuda(tokenized, model_inputs, generated)
    return clean_hy_translation(out)


def load_translategemma(
    model_name: str = DEFAULT_TRANSLATEGEMMA_MODEL,
    device: str = "cuda",
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = pick_dtype(torch)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" and torch.cuda.is_available() else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return {"model": model, "processor": processor, "model_id": model_name, "dtype": str(dtype)}


def translate_with_translategemma(
    state: Dict[str, Any],
    text: str,
    use_terminology: bool = False,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    import torch

    text = normalize_source_text(text)
    if not text:
        return ""

    processor, model = state["processor"], state["model"]
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "source_lang_code": "ru", "target_lang_code": "en", "text": text}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(get_input_device(model))

    input_len = inputs["input_ids"].shape[1]
    tok = processor.tokenizer
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tok.eos_token_id]
    if eot_id is not None and eot_id != tok.unk_token_id and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_ids,
            pad_token_id=tok.eos_token_id,
        )

    out = processor.decode(generated[0][input_len:], skip_special_tokens=True)
    cleanup_cuda(inputs, generated)
    return clean_translation(out)


TRANSLATOR_REGISTRY = {
    "qwen": {
        "loader": load_qwen,
        "translator": translate_with_qwen,
        "default_model": DEFAULT_QWEN_MODEL,
    },
    "hy_mt": {
        "loader": load_hy_mt,
        "translator": translate_with_hy_mt,
        "default_model": DEFAULT_HY_MT_MODEL,
    },
    "translategemma": {
        "loader": load_translategemma,
        "translator": translate_with_translategemma,
        "default_model": DEFAULT_TRANSLATEGEMMA_MODEL,
    },
}


def _model_name_for_key(key: str, model_names: Optional[Dict[str, str]]) -> str:
    if model_names and key in model_names:
        return model_names[key]
    return TRANSLATOR_REGISTRY[key]["default_model"]


def _translate_pair_lists(
    refs: Sequence[str],
    cands: Sequence[str],
    translator: Any,
    state: Dict[str, Any],
    use_terms: bool,
    max_new_tokens: int,
) -> Tuple[List[str], List[str]]:
    cache: Dict[Tuple[str, bool], str] = {}
    gt_vals: List[str] = []
    gen_vals: List[str] = []

    for ref_text, cand_text in zip(refs, cands):
        for src, target_list in ((ref_text, gt_vals), (cand_text, gen_vals)):
            normalized = normalize_source_text(src)
            key = (normalized, use_terms)
            if key not in cache:
                cache[key] = translator(
                    state,
                    normalized,
                    use_terminology=use_terms,
                    max_new_tokens=max_new_tokens,
                )
            target_list.append(cache[key])

    return gt_vals, gen_vals


def _translate_one_translator_columns(
    input_csv: str,
    tr_key: str,
    candidate_col: str,
    reference_col: str,
    kinds: Sequence[str],
    model_names: Optional[Dict[str, str]],
    gpu: str,
    device: str,
    max_new_tokens: int,
) -> Dict[str, List[str]]:
    configure_worker_environment(gpu=gpu, device=device)

    if tr_key not in TRANSLATOR_REGISTRY:
        raise ValueError(f"Unknown translator: {tr_key}")

    df = pd.read_csv(input_csv)
    refs = df[reference_col].fillna("").astype(str).tolist()
    cands = df[candidate_col].fillna("").astype(str).tolist()

    loader = TRANSLATOR_REGISTRY[tr_key]["loader"]
    translator = TRANSLATOR_REGISTRY[tr_key]["translator"]
    state = loader(_model_name_for_key(tr_key, model_names), device=device)

    out: Dict[str, List[str]] = {}

    try:
        if tr_key == "translategemma":
            if any(kind in kinds for kind in ("terms", "noterms")):
                gt_vals, gen_vals = _translate_pair_lists(
                    refs=refs,
                    cands=cands,
                    translator=translator,
                    state=state,
                    use_terms=False,
                    max_new_tokens=max_new_tokens,
                )
                if "noterms" in kinds:
                    out[f"gt_{tr_key}_noterms"] = gt_vals
                    out[f"generation_{tr_key}_noterms"] = gen_vals
                if "terms" in kinds:
                    out[f"gt_{tr_key}_terms"] = list(gt_vals)
                    out[f"generation_{tr_key}_terms"] = list(gen_vals)

        else:
            for kind in kinds:
                if kind not in {"terms", "noterms"}:
                    raise ValueError(f"Unknown translation kind: {kind}. Expected terms/noterms.")

                use_terms = kind == "terms"
                gt_vals, gen_vals = _translate_pair_lists(
                    refs=refs,
                    cands=cands,
                    translator=translator,
                    state=state,
                    use_terms=use_terms,
                    max_new_tokens=max_new_tokens,
                )
                out[f"gt_{tr_key}_{kind}"] = gt_vals
                out[f"generation_{tr_key}_{kind}"] = gen_vals

        return out

    finally:
        cleanup_cuda(state)


def _translate_one_translator_worker(
    result_queue: Any,
    worker_kwargs: Dict[str, Any],
) -> None:
    tr_key = str(worker_kwargs.get("tr_key", "UNKNOWN"))

    try:
        columns = _translate_one_translator_columns(**worker_kwargs)
        result_queue.put(
            {
                "ok": True,
                "tr_key": tr_key,
                "columns": columns,
            }
        )
    except BaseException as e:
        result_queue.put(
            {
                "ok": False,
                "tr_key": tr_key,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        )


def _run_one_translator_in_separate_process(
    worker_kwargs: Dict[str, Any],
) -> Dict[str, List[str]]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_translate_one_translator_worker,
        args=(result_queue, worker_kwargs),
    )

    tr_key = str(worker_kwargs.get("tr_key", "UNKNOWN"))
    proc.start()

    payload: Optional[Dict[str, Any]] = None
    try:
        while True:
            try:
                payload = result_queue.get(timeout=1.0)
                break
            except queue.Empty:
                if not proc.is_alive():
                    proc.join()
                    raise RuntimeError(
                        f"Translator worker '{tr_key}' exited with code {proc.exitcode} "
                        "without returning a result."
                    )

        proc.join()

        if proc.exitcode != 0 and payload.get("ok", False):
            raise RuntimeError(
                f"Translator worker '{tr_key}' returned data but exited with non-zero code {proc.exitcode}."
            )

        if not payload.get("ok", False):
            raise RuntimeError(
                f"Translator worker '{tr_key}' failed:\n"
                f"{payload.get('error')}\n\n"
                f"{payload.get('traceback', '')}"
            )

        columns = payload.get("columns")
        if not isinstance(columns, dict):
            raise RuntimeError(f"Translator worker '{tr_key}' returned invalid columns payload.")

        return columns

    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()

        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass


def translate_reports_for_csv(
    input_csv: str,
    output_csv: str,
    candidate_col: str = CANDIDATE_COL,
    reference_col: str = REFERENCE_COL,
    row_id_col: str = ROW_ID_COL,
    translators: Optional[Sequence[str]] = None,
    kinds: Optional[Sequence[str]] = None,
    model_names: Optional[Dict[str, str]] = None,
    gpu: str = "0",
    device: str = "cuda",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    configure_worker_environment(gpu=gpu, device=device)

    translators = list(translators or ["qwen", "hy_mt", "translategemma"])
    kinds = list(kinds or ["terms", "noterms"])

    unknown = sorted(set(translators) - set(TRANSLATOR_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown translators: {unknown}. Available: {sorted(TRANSLATOR_REGISTRY)}")

    unknown_kinds = sorted(set(kinds) - {"terms", "noterms"})
    if unknown_kinds:
        raise ValueError(f"Unknown translation kinds: {unknown_kinds}. Available: terms, noterms")

    df = pd.read_csv(input_csv)
    out = pd.DataFrame(
        {
            row_id_col: (
                df[row_id_col].astype(str).tolist()
                if row_id_col in df.columns
                else [str(i) for i in range(len(df))]
            )
        }
    )

    for tr_key in translators:
        worker_kwargs = {
            "input_csv": input_csv,
            "tr_key": tr_key,
            "candidate_col": candidate_col,
            "reference_col": reference_col,
            "kinds": kinds,
            "model_names": model_names,
            "gpu": gpu,
            "device": device,
            "max_new_tokens": max_new_tokens,
        }
        columns = _run_one_translator_in_separate_process(worker_kwargs)

        for col_name, values in columns.items():
            if len(values) != len(out):
                raise RuntimeError(
                    f"Translator '{tr_key}' returned column '{col_name}' with length {len(values)}, "
                    f"expected {len(out)}."
                )
            out[col_name] = values

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate Russian radiology reports to English.")
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--candidate-col", default=CANDIDATE_COL)
    p.add_argument("--reference-col", default=REFERENCE_COL)
    p.add_argument("--row-id-col", default=ROW_ID_COL)
    p.add_argument("--translators", default="qwen,hy_mt,translategemma")
    p.add_argument("--kinds", default="terms,noterms")
    p.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    p.add_argument("--hy-mt-model", default=DEFAULT_HY_MT_MODEL)
    p.add_argument("--translategemma-model", default=DEFAULT_TRANSLATEGEMMA_MODEL)
    p.add_argument("--gpu", default="0")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_names = {
        "qwen": args.qwen_model,
        "hy_mt": args.hy_mt_model,
        "translategemma": args.translategemma_model,
    }

    translate_reports_for_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        candidate_col=args.candidate_col,
        reference_col=args.reference_col,
        row_id_col=args.row_id_col,
        translators=split_csv_arg(args.translators, ["qwen", "hy_mt", "translategemma"]),
        kinds=split_csv_arg(args.kinds, ["terms", "noterms"]),
        model_names=model_names,
        gpu=args.gpu,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
