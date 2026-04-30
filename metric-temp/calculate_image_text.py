from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import IMAGE_COL, ROW_ID_COL, cleanup_cuda, configure_worker_environment, get_input_device, require_column

DEFAULT_BIOVILT_MODEL = "microsoft/BiomedVLP-BioViL-T"


def safe_texts(values: Sequence[Any]) -> List[str]:
    return ["" if pd.isna(v) else str(v) for v in values]


def resolve_image_paths(paths: Sequence[Any], image_root: Optional[str]) -> List[str]:
    root = Path(image_root) if image_root else None
    resolved = []
    for path in paths:
        p = Path(str(path))
        if root is not None and not p.is_absolute():
            p = root / p
        resolved.append(str(p))
    return resolved


def pair_columns(df: pd.DataFrame, translators: Sequence[str], kinds: Sequence[str]) -> List[Tuple[str, str, str]]:
    pairs = []
    for translator in translators:
        for kind in kinds:
            col = f"generation_{translator}_{kind}"
            require_column(df, col)
            pairs.append((translator, kind, col))
    return pairs


def l2_tensor(x: Any):
    import torch.nn.functional as F

    return F.normalize(x, dim=-1)


def load_biovilt_text_encoder(model_name: str, device: Optional[str]):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.tie_word_embeddings = False
    model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False, device_map=None)
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    actual_device = get_input_device(device)
    actual_device = "cuda" if actual_device != "cpu" and torch.cuda.is_available() else "cpu"
    model.to(actual_device)
    model.eval()
    return tokenizer, model


def encode_biovilt_text(texts: Sequence[str], tokenizer: Any, model: Any, batch_size: int = 16) -> np.ndarray:
    import torch

    embs = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(list(texts[start : start + batch_size]), padding=True, truncation=True, max_length=512, return_tensors="pt")
        batch.pop("token_type_ids", None)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.inference_mode():
            if hasattr(model, "get_projected_text_embeddings"):
                emb = model.get_projected_text_embeddings(**batch)
            else:
                output = model(**batch, return_dict=True)
                emb = output.last_hidden_state[:, 0, :]
            emb = l2_tensor(emb)
        embs.append(emb.detach().cpu().numpy())
    cleanup_cuda()
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), dtype=np.float32)


def encode_biovilt_images(paths: Sequence[str], device: Optional[str]) -> np.ndarray:
    import torch
    from health_multimodal.image.utils import ImageModelType, get_image_inference

    engine = get_image_inference(ImageModelType.BIOVIL_T)
    actual_device = get_input_device(device)
    if hasattr(engine, "model"):
        if actual_device != "cpu" and torch.cuda.is_available():
            engine.model.to("cuda")
        else:
            engine.model.to("cpu")
    embs = []
    for path in paths:
        with torch.inference_mode():
            emb = engine.get_projected_global_embedding(Path(path))
            emb = l2_tensor(emb.unsqueeze(0) if emb.ndim == 1 else emb)
        embs.append(emb.detach().cpu().numpy()[0])
    cleanup_cuda(engine)
    return np.stack(embs, axis=0) if embs else np.zeros((0, 1), dtype=np.float32)


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


def calculate_image_text_metrics_for_csv(
    input_csv: str,
    output_csv: str,
    translators: Optional[Sequence[str]] = None,
    translation_kinds: Optional[Sequence[str]] = None,
    image_root: Optional[str] = None,
    biovilt_model: str = DEFAULT_BIOVILT_MODEL,
    batch_size: int = 16,
    device: Optional[str] = None,
    gpu: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    configure_worker_environment(gpu)
    translators = list(translators or ["qwen", "hy_mt", "translategemma"])
    translation_kinds = list(translation_kinds or ["terms", "noterms"])
    df = pd.read_csv(input_csv)
    for col in (ROW_ID_COL, IMAGE_COL):
        require_column(df, col)
    pairs = pair_columns(df, translators, translation_kinds)
    paths = resolve_image_paths(df[IMAGE_COL].tolist(), image_root)
    image_emb = encode_biovilt_images(paths, device=device)
    tokenizer, model = load_biovilt_text_encoder(biovilt_model, device=device)
    out = pd.DataFrame({ROW_ID_COL: df[ROW_ID_COL]})
    for translator, kind, text_col in pairs:
        text_emb = encode_biovilt_text(safe_texts(df[text_col]), tokenizer, model, batch_size=batch_size)
        out[f"img_en_biovilt_{translator}_{kind}"] = cosine_scores(image_emb, text_emb)
    out.to_csv(output_csv, index=False)
    cleanup_cuda(model, tokenizer)
    return out
