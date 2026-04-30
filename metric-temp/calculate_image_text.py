from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import IMAGE_COL, ROW_ID_COL, cleanup_cuda, configure_worker_environment, get_input_device, split_csv_arg

DEFAULT_BIOVILT_MODEL = "microsoft/BiomedVLP-BioViL-T"


def safe_texts(xs: Sequence[Any]) -> List[str]:
    return ["" if pd.isna(x) else str(x) for x in xs]


def resolve_image_paths(paths: Sequence[str], image_root: Optional[str]) -> List[str]:
    root = Path(image_root).expanduser().resolve() if image_root else None
    out: List[str] = []
    for p in paths:
        pp = Path(str(p))
        if not pp.is_absolute() and root is not None:
            pp = root / pp
        out.append(str(pp))
    return out


def pair_columns(df: pd.DataFrame, translators: Sequence[str], kinds: Sequence[str]) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    for tr in translators:
        for kind in kinds:
            gen_col = f"generation_{tr}_{kind}"
            if gen_col in df.columns:
                pairs.append((tr, kind, gen_col))
    if not pairs:
        raise ValueError("No candidate translation columns found. Expected generation_qwen_noterms etc.")
    return pairs


def l2_tensor(x: Any) -> Any:
    import torch.nn.functional as F

    return F.normalize(x.float(), p=2, dim=-1)


def load_biovilt_text_encoder(model_name: str, device: str) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    actual_device = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.tie_word_embeddings = False
    model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False, device_map=None).to(actual_device)
    model.eval()
    return tokenizer, model


def encode_biovilt_text(texts: Sequence[str], tokenizer: Any, model: Any, batch_size: int = 16) -> Any:
    import torch

    device = get_input_device(model)
    xs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(
                list(texts[i : i + batch_size]),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            if hasattr(model, "get_projected_text_embeddings"):
                emb = model.get_projected_text_embeddings(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            else:
                emb = model(**batch, return_dict=True).last_hidden_state[:, 0, :]
            xs.append(l2_tensor(emb).detach().cpu().float())
            cleanup_cuda(batch, emb)
    return torch.cat(xs, dim=0)


def encode_biovilt_images(paths: Sequence[str], batch_size: int = 8) -> Any:
    import torch
    from health_multimodal.image.utils import ImageModelType, get_image_inference

    engine = get_image_inference(ImageModelType.BIOVIL_T)
    xs = []
    for i in range(0, len(paths), batch_size):
        ys = []
        for p in paths[i : i + batch_size]:
            y = engine.get_projected_global_embedding(Path(p))
            if y.ndim == 2 and y.shape[0] == 1:
                y = y.squeeze(0)
            ys.append(l2_tensor(y).detach().cpu().float())
        xs.append(torch.stack(ys, dim=0))
    cleanup_cuda(engine)
    return torch.cat(xs, dim=0)


def cosine_scores(a: Any, b: Any) -> np.ndarray:
    import torch.nn.functional as F

    return F.cosine_similarity(l2_tensor(a), l2_tensor(b), dim=-1).cpu().numpy()


def calculate_image_text_metrics_for_csv(
    input_csv: str,
    output_csv: str,
    image_col: str = IMAGE_COL,
    row_id_col: str = ROW_ID_COL,
    translators: Optional[Sequence[str]] = None,
    kinds: Optional[Sequence[str]] = None,
    image_root: Optional[str] = None,
    biovilt_model: str = DEFAULT_BIOVILT_MODEL,
    batch_size: int = 8,
    gpu: str = "0",
    device: str = "cuda",
) -> str:
    configure_worker_environment(gpu=gpu, device=device)
    translators = list(translators or ["qwen", "hy_mt", "translategemma"])
    kinds = list(kinds or ["terms", "noterms"])
    df = pd.read_csv(input_csv)
    if image_col not in df.columns:
        raise ValueError(f"Missing image column: {image_col}")
    pairs = pair_columns(df, translators, kinds)
    image_paths = resolve_image_paths(df[image_col].fillna("").astype(str).tolist(), image_root=image_root)

    image_emb = encode_biovilt_images(image_paths, batch_size=batch_size)
    tokenizer, model = load_biovilt_text_encoder(biovilt_model, device=device)
    try:
        out = pd.DataFrame({row_id_col: df[row_id_col].astype(str).tolist() if row_id_col in df.columns else [str(i) for i in range(len(df))]})
        for tr, kind, gen_col in pairs:
            texts = safe_texts(df[gen_col].tolist())
            text_emb = encode_biovilt_text(texts, tokenizer, model, batch_size=max(1, batch_size * 2))
            out[f"img_en_biovilt_{tr}_{kind}"] = cosine_scores(image_emb, text_emb)
            cleanup_cuda(text_emb)
    finally:
        cleanup_cuda(tokenizer, model, image_emb)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calculate BioViL-T image-text cosine similarity only.")
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--image-col", default=IMAGE_COL)
    p.add_argument("--row-id-col", default=ROW_ID_COL)
    p.add_argument("--translators", default="qwen,hy_mt,translategemma")
    p.add_argument("--kinds", default="terms,noterms")
    p.add_argument("--image-root", default=None)
    p.add_argument("--biovilt-model", default=DEFAULT_BIOVILT_MODEL)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gpu", default="0")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    calculate_image_text_metrics_for_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        image_col=args.image_col,
        row_id_col=args.row_id_col,
        translators=split_csv_arg(args.translators, []),
        kinds=split_csv_arg(args.kinds, []),
        image_root=args.image_root,
        biovilt_model=args.biovilt_model,
        batch_size=args.batch_size,
        gpu=args.gpu,
        device=args.device,
    )


if __name__ == "__main__":
    main()
