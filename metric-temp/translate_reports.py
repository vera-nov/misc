from __future__ import annotations

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
    require_column,
)

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


def normalize_source_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def remove_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def maybe_extract_json_translation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            for key in ("translation", "english", "text", "output"):
                if key in obj and isinstance(obj[key], str):
                    return obj[key]
        if isinstance(obj, str):
            return obj
    except Exception:
        return stripped
    return stripped


def clean_translation(text: str) -> str:
    text = remove_think_blocks(text)
    text = maybe_extract_json_translation(text)
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"^English translation\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Translation\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def safe_chat_template(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "\n".join(m["content"] for m in messages) + "\n"


def load_qwen(model_name: str, device: Optional[str]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    actual_device = get_input_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": pick_dtype(),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if actual_device != "cpu" and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if actual_device == "cpu":
        model.to("cpu")
    model.eval()
    return tokenizer, model


def build_qwen_prompt(text: str, use_terminology: bool) -> str:
    base = (
        "You are a medical translator.\n"
        "Translate the following chest X-ray report from Russian to English.\n"
        "Return only the English translation.\n"
        "Do not add explanations, comments, bullet points, or reasoning.\n"
    )
    if use_terminology:
        base += "If one of the following Russian medical terms appears, use the specified English equivalent exactly.\n\n" + GLOSSARY_TEXT_RU_EN + "\n\n"
    else:
        base += "\n"
    return base + text


def translate_with_qwen(texts: Sequence[str], model_name: str, use_terminology: bool, max_new_tokens: int, device: Optional[str]) -> List[str]:
    import torch

    tokenizer, model = load_qwen(model_name, device)
    outputs: List[str] = []
    for text in texts:
        src = normalize_source_text(text)
        if not src:
            outputs.append("")
            continue
        prompt = build_qwen_prompt(src, use_terminology)
        messages = [
            {"role": "system", "content": "You are a precise medical translator. Return only the English translation."},
            {"role": "user", "content": prompt},
        ]
        rendered = safe_chat_template(tokenizer, messages)
        batch = tokenizer([rendered], return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][batch["input_ids"].shape[1] :]
        outputs.append(clean_translation(tokenizer.decode(new_tokens, skip_special_tokens=True)))
    cleanup_cuda(model, tokenizer)
    return outputs


def load_hy_mt(model_name: str, device: Optional[str]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    actual_device = get_input_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": pick_dtype(),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if actual_device != "cpu" and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if actual_device == "cpu":
        model.to("cpu")
    model.eval()
    return tokenizer, model


def format_glossary_for_hy_mt(glossary_pairs: Sequence[Tuple[str, str]]) -> str:
    return "\n".join(f"{src} translates as {tgt}" for tgt, src in glossary_pairs)


def build_hy_prompt(text: str, target_language: str, use_terminology: bool) -> str:
    if use_terminology:
        glossary_block = format_glossary_for_hy_mt(GLOSSARY)
        return f"Refer to the following translations:\n{glossary_block}\n\nTranslate the following segment into {target_language}, without additional explanation.\n\n{text}"
    return f"Translate the following segment into {target_language}, without additional explanation.\n\n{text}"


def clean_hy_translation(text: str) -> str:
    return clean_translation(text)


def translate_with_hy_mt(texts: Sequence[str], model_name: str, use_terminology: bool, max_new_tokens: int, device: Optional[str]) -> List[str]:
    import torch

    tokenizer, model = load_hy_mt(model_name, device)
    outputs: List[str] = []
    for text in texts:
        src = normalize_source_text(text)
        if not src:
            outputs.append("")
            continue
        prompt = build_hy_prompt(src, "English", use_terminology)
        batch = tokenizer(prompt, return_tensors="pt", truncation=True)
        batch.pop("token_type_ids", None)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][batch["input_ids"].shape[1] :]
        outputs.append(clean_hy_translation(tokenizer.decode(new_tokens, skip_special_tokens=True)))
    cleanup_cuda(model, tokenizer)
    return outputs


def load_translategemma(model_name: str, device: Optional[str]):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    actual_device = get_input_device(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": pick_dtype(),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if actual_device != "cpu" and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if actual_device == "cpu":
        model.to("cpu")
    model.eval()
    return processor, model


def translate_with_translategemma(texts: Sequence[str], model_name: str, use_terminology: bool, max_new_tokens: int, device: Optional[str]) -> List[str]:
    import torch

    processor, model = load_translategemma(model_name, device)
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_ids = [tokenizer.eos_token_id]
    try:
        end_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_turn, int) and end_turn >= 0:
            eos_ids.append(end_turn)
    except Exception:
        pass
    outputs: List[str] = []
    for text in texts:
        src = normalize_source_text(text)
        if not src:
            outputs.append("")
            continue
        messages = [{"role": "user", "content": [{"type": "text", "source_lang_code": "ru", "target_lang_code": "en", "text": src}]}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs.append(clean_translation(tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)))
    cleanup_cuda(model, processor)
    return outputs


TRANSLATOR_REGISTRY = {
    "qwen": translate_with_qwen,
    "hy_mt": translate_with_hy_mt,
    "translategemma": translate_with_translategemma,
}


def _model_name_for_key(key: str, qwen_model: str, hy_mt_model: str, translategemma_model: str) -> str:
    return {
        "qwen": qwen_model,
        "hy_mt": hy_mt_model,
        "translategemma": translategemma_model,
    }[key]


def _empty_translation_indices(texts: Sequence[str], outputs: Sequence[str]) -> List[int]:
    return [
        i
        for i, (src, out) in enumerate(zip(texts, outputs))
        if normalize_source_text(src) and not normalize_source_text(out)
    ]


def _run_translator(
    texts: Sequence[str],
    translator: str,
    model_name: str,
    use_terminology: bool,
    max_new_tokens: int,
    device: Optional[str],
    label: str,
) -> List[str]:
    outputs = TRANSLATOR_REGISTRY[translator](texts, model_name, use_terminology, max_new_tokens, device)
    if len(outputs) != len(texts):
        raise ValueError(
            f"Translator {translator} returned {len(outputs)} translations for {len(texts)} inputs in {label}"
        )
    return list(outputs)


def _translate_pair_lists(
    texts: Sequence[str],
    translator: str,
    model_name: str,
    use_terminology: bool,
    max_new_tokens: int,
    device: Optional[str],
    label: str,
    retry_empty_translations: int = 2,
) -> List[str]:
    texts = list(texts)
    outputs = _run_translator(texts, translator, model_name, use_terminology, max_new_tokens, device, label)

    for _ in range(max(0, retry_empty_translations)):
        bad = _empty_translation_indices(texts, outputs)
        if not bad:
            break
        retry_texts = [texts[i] for i in bad]
        retry_outputs = _run_translator(
            retry_texts,
            translator,
            model_name,
            use_terminology,
            max_new_tokens,
            device,
            label,
        )
        for i, value in zip(bad, retry_outputs):
            outputs[i] = value

    bad = _empty_translation_indices(texts, outputs)
    if bad:
        fallback_texts = [texts[i] for i in bad]
        fallback_outputs = _run_translator(
            fallback_texts,
            translator,
            model_name,
            not use_terminology,
            max_new_tokens,
            device,
            f"{label}/fallback_terms_{str(not use_terminology).lower()}",
        )
        for i, value in zip(bad, fallback_outputs):
            outputs[i] = value

    bad = _empty_translation_indices(texts, outputs)
    if bad:
        raise ValueError(
            f"Translator {translator} produced empty translations in {label} "
            f"for row indices: {bad[:20]}"
        )

    return outputs


def _translate_one_translator_columns(
    df: pd.DataFrame,
    translator: str,
    kinds: Sequence[str],
    model_name: str,
    max_new_tokens: int,
    device: Optional[str],
    retry_empty_translations: int = 2,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    candidate_texts = df[CANDIDATE_COL].map(normalize_source_text).tolist()
    reference_texts = df[REFERENCE_COL].map(normalize_source_text).tolist()

    for kind in kinds:
        if kind not in {"terms", "noterms"}:
            raise ValueError(f"Unknown translation kind: {kind}")

    if translator == "translategemma":
        cand = _translate_pair_lists(
            candidate_texts,
            translator,
            model_name,
            False,
            max_new_tokens,
            device,
            f"{translator}/generation/noterms",
            retry_empty_translations,
        )
        ref = _translate_pair_lists(
            reference_texts,
            translator,
            model_name,
            False,
            max_new_tokens,
            device,
            f"{translator}/gt/noterms",
            retry_empty_translations,
        )
        for kind in kinds:
            result[f"generation_{translator}_{kind}"] = cand
            result[f"gt_{translator}_{kind}"] = ref
        return result

    for kind in kinds:
        use_terms = kind == "terms"
        result[f"generation_{translator}_{kind}"] = _translate_pair_lists(
            candidate_texts,
            translator,
            model_name,
            use_terms,
            max_new_tokens,
            device,
            f"{translator}/generation/{kind}",
            retry_empty_translations,
        )
        result[f"gt_{translator}_{kind}"] = _translate_pair_lists(
            reference_texts,
            translator,
            model_name,
            use_terms,
            max_new_tokens,
            device,
            f"{translator}/gt/{kind}",
            retry_empty_translations,
        )
    return result


def _translate_one_translator_worker(payload: Dict[str, Any], out_queue: mp.Queue) -> None:
    try:
        configure_worker_environment(payload.get("gpu"))
        df = pd.read_csv(payload["input_csv"])
        for col in (ROW_ID_COL, CANDIDATE_COL, REFERENCE_COL):
            require_column(df, col)
        cols = _translate_one_translator_columns(
            df=df,
            translator=payload["translator"],
            kinds=payload["kinds"],
            model_name=payload["model_name"],
            max_new_tokens=payload["max_new_tokens"],
            device=payload.get("device"),
            retry_empty_translations=payload.get("retry_empty_translations", 2),
        )
        out = pd.DataFrame({ROW_ID_COL: df[ROW_ID_COL]})
        for k, v in cols.items():
            out[k] = v
        out_csv = payload["out_csv"]
        out.to_csv(out_csv, index=False)
        out_queue.put({"ok": True, "out_csv": out_csv})
    except Exception:
        out_queue.put({"ok": False, "error": traceback.format_exc()})

def _run_one_translator_in_separate_process(payload: Dict[str, Any], timeout: int) -> pd.DataFrame:
    ctx = mp.get_context("spawn")
    out_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_translate_one_translator_worker, args=(payload, out_queue))
    proc.start()
    try:
        message = out_queue.get(timeout=timeout)
    except queue.Empty:
        proc.terminate()
        proc.join(timeout=10)
        raise TimeoutError(f"Translator {payload['translator']} timed out after {timeout} seconds")
    finally:
        if proc.is_alive():
            proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
    if not message.get("ok"):
        raise RuntimeError(message.get("error", "translation worker failed"))
    return pd.read_csv(message["out_csv"])


def translate_reports_for_csv(
    input_csv: str,
    output_csv: str,
    translators: Optional[Sequence[str]] = None,
    translation_kinds: Optional[Sequence[str]] = None,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    hy_mt_model: str = DEFAULT_HY_MT_MODEL,
    translategemma_model: str = DEFAULT_TRANSLATEGEMMA_MODEL,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: Optional[str] = None,
    gpu: Optional[int] = None,
    per_translator_timeout: int = 3600,
    retry_empty_translations: int = 2,
    verbose: bool = False,
) -> pd.DataFrame:
    configure_worker_environment(gpu)
    translators = list(translators or ["qwen", "hy_mt", "translategemma"])
    translation_kinds = list(translation_kinds or ["terms", "noterms"])
    for translator in translators:
        if translator not in TRANSLATOR_REGISTRY:
            raise ValueError(f"Unknown translator: {translator}")
    df = pd.read_csv(input_csv)
    for col in (ROW_ID_COL, CANDIDATE_COL, REFERENCE_COL):
        require_column(df, col)
    out = pd.DataFrame({ROW_ID_COL: df[ROW_ID_COL]})
    tmp_dir = Path(output_csv).resolve().parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_copy = tmp_dir / (Path(output_csv).stem + ".translation_input.csv")
    df.to_csv(input_copy, index=False)
    for translator in translators:
        payload = {
            "input_csv": str(input_copy),
            "out_csv": str(tmp_dir / f"{Path(output_csv).stem}.{translator}.csv"),
            "translator": translator,
            "kinds": translation_kinds,
            "model_name": _model_name_for_key(translator, qwen_model, hy_mt_model, translategemma_model),
            "max_new_tokens": max_new_tokens,
            "device": device,
            "gpu": gpu,
            "retry_empty_translations": retry_empty_translations,
        }
        if verbose:
            print(f"Running translator: {translator}", flush=True)
        chunk = _run_one_translator_in_separate_process(payload, per_translator_timeout)
        if len(chunk) != len(df):
            raise ValueError(f"Translator {translator} returned {len(chunk)} rows for {len(df)} inputs")
        out = out.merge(chunk, on=ROW_ID_COL, how="left")
    out.to_csv(output_csv, index=False)
    cleanup_cuda()
    return out
