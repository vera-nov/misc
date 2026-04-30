from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import (
    CANDIDATE_COL,
    REFERENCE_COL,
    ROW_ID_COL,
    cleanup_cuda,
    configure_worker_environment,
    get_input_device,
    pick_dtype,
)

PROMPT_WORD_LIMIT = 300
MAX_MODEL_LENGTH = 2048
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def safe_apply_qwen_chat_template(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    for kwargs in (
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {},
    ):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except TypeError:
            continue
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def strip_thinking(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


@dataclass
class ParsedGreen:
    score: float
    sig_total: Optional[int]
    insig_total: Optional[int]
    matched_findings: Optional[int]
    errors: List[Dict[str, Any]]


class ManualGREENQwen:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 1,
        prompt_word_limit: int = PROMPT_WORD_LIMIT,
        max_model_length: int = MAX_MODEL_LENGTH,
        max_new_tokens: int = 768,
        device: str = "cuda",
    ) -> None:
        warnings.filterwarnings("ignore")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.prompt_word_limit = prompt_word_limit
        self.max_model_length = max_model_length
        self.max_new_tokens = max_new_tokens
        self.device_policy = device

        dtype = pick_dtype(torch)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if device != "cpu" and torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self.input_device = get_input_device(self.model)

    def make_prompt(self, reference_text: str, candidate_text: str) -> str:
        reference_text = " ".join(str(reference_text).split()[: self.prompt_word_limit])
        candidate_text = " ".join(str(candidate_text).split()[: self.prompt_word_limit])
        return f"""Цель: Оценить точность проверяемого радиологического отчета по сравнению с эталонным радиологическим отчетом, написанным экспертами-радиологами.

Обзор процесса:
Ты получишь:
1. Критерии оценки.
2. Эталонный радиологический отчет.
3. Проверяемый радиологический отчет.
4. Требуемый формат ответа.

1. Критерии оценки:
Для проверяемого отчета определи:
- количество клинически значимых ошибок
- количество клинически незначимых ошибок

Возможные категории ошибок:
(а) Ложное указание находки в проверяемом отчете
(б) Пропуск находки, присутствующей в эталонном отчете
(в) Неверное определение анатомической локализации или положения находки
(г) Неверная оценка степени выраженности находки
(д) Упоминание сравнения, которого нет в эталонном отчете
(е) Пропуск сравнения, описывающего изменение по сравнению с предыдущим исследованием

Сосредоточься на клинических находках, а не на стиле изложения. Оценивай только находки, которые присутствуют в отчетах.

2. Эталонный отчет:
{reference_text}

3. Проверяемый отчет:
{candidate_text}

4. Представление оценки:
Строго следуй этому формату, даже если ошибок не найдено:

[Объяснение]:
<краткое объяснение>

[Клинически значимые ошибки]:
(а) Ложное указание находки в проверяемом отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(б) Пропуск находки, присутствующей в эталонном отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(в) Неверное определение анатомической локализации или положения находки: <число>. <ошибка 1>; <ошибка 2>; ...
(г) Неверная оценка степени выраженности находки: <число>. <ошибка 1>; <ошибка 2>; ...
(д) Упоминание сравнения, которого нет в эталонном отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(е) Пропуск сравнения, описывающего изменение по сравнению с предыдущим исследованием: <число>. <ошибка 1>; <ошибка 2>; ...

[Клинически незначимые ошибки]:
(а) Ложное указание находки в проверяемом отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(б) Пропуск находки, присутствующей в эталонном отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(в) Неверное определение анатомической локализации или положения находки: <число>. <ошибка 1>; <ошибка 2>; ...
(г) Неверная оценка степени выраженности находки: <число>. <ошибка 1>; <ошибка 2>; ...
(д) Упоминание сравнения, которого нет в эталонном отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(е) Пропуск сравнения, описывающего изменение по сравнению с предыдущим исследованием: <число>. <ошибка 1>; <ошибка 2>; ...

[Совпадающие находки]:
<число>. <находка 1>; <находка 2>; ...

Если во всем разделе клинически значимых ошибок нет ошибок, напиши:
Клинически значимых ошибок нет.

Если во всем разделе клинически незначимых ошибок нет ошибок, напиши:
Клинически незначимых ошибок нет.
"""

    def clean_response(self, response: str) -> str:
        response = strip_thinking(response)
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        response = response.strip()
        response = re.sub(r"^```[a-zA-Z]*\n?", "", response)
        response = re.sub(r"\n?```$", "", response)
        if "[Объяснение]:" in response:
            response = response[response.find("[Объяснение]:") :]
        return response.strip()

    def _tokenize_batch(self, prompts: Sequence[str]) -> Dict[str, Any]:
        rendered = [safe_apply_qwen_chat_template(self.tokenizer, [{"role": "user", "content": p}]) for p in prompts]
        batch = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_model_length,
            add_special_tokens=False,
        )
        return {k: v.to(self.input_device) for k, v in batch.items()}

    def generate_batch(self, prompts: Sequence[str]) -> List[str]:
        torch = self.torch
        tokenized = self._tokenize_batch(prompts)
        input_len = tokenized["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **tokenized,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[:, input_len:]
        responses = self.tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        cleanup_cuda(tokenized, outputs, generated)
        return [self.clean_response(r) for r in responses]

    @staticmethod
    def _extract_section(text: str, title: str) -> Optional[str]:
        titles = [
            "Клинически значимые ошибки",
            "Клинически незначимые ошибки",
            "Совпадающие находки",
        ]
        next_titles = [t for t in titles if t != title]
        start = re.search(rf"\[{re.escape(title)}\]:\s*", text, flags=re.IGNORECASE)
        if not start:
            return None
        end_pos = len(text)
        for nt in next_titles:
            m = re.search(rf"\n\s*\[{re.escape(nt)}\]:", text[start.end():], flags=re.IGNORECASE)
            if m:
                end_pos = min(end_pos, start.end() + m.start())
        return text[start.end():end_pos].strip()

    @staticmethod
    def _parse_count_and_desc(line: str) -> Tuple[int, str]:
        m = re.search(r":\s*(\d+)\s*\.\s*(.*)$", line.strip())
        if not m:
            return 0, ""
        return int(m.group(1)), m.group(2).strip()

    def parse(self, text: str) -> ParsedGreen:
        errors: List[Dict[str, Any]] = []
        totals: Dict[str, Optional[int]] = {"significant": None, "insignificant": None}
        subcats = {
            "а": "Ложное указание находки в проверяемом отчете",
            "б": "Пропуск находки, присутствующей в эталонном отчете",
            "в": "Неверное определение анатомической локализации или положения находки",
            "г": "Неверная оценка степени выраженности находки",
            "д": "Упоминание сравнения, которого нет в эталонном отчете",
            "е": "Пропуск сравнения, описывающего изменение по сравнению с предыдущим исследованием",
        }
        for title, key in [("Клинически значимые ошибки", "significant"), ("Клинически незначимые ошибки", "insignificant")]:
            block = self._extract_section(text, title)
            if block is None:
                totals[key] = None
                continue
            block_l = block.lower().strip()
            if (
                not block
                or block_l.startswith("клинически значимых ошибок нет")
                or block_l.startswith("клинически незначимых ошибок нет")
                or block_l.startswith("ошибок нет")
            ):
                totals[key] = 0
                continue
            total = 0
            for letter, subcat in subcats.items():
                line_match = re.search(rf"^\s*\({letter}\)\s*.*$", block, flags=re.MULTILINE)
                if not line_match:
                    continue
                count, desc = self._parse_count_and_desc(line_match.group(0))
                total += count
                if count > 0:
                    pieces = [p.strip() for p in re.split(r";|\n", desc) if p.strip() and p.strip() != "..."]
                    errors.append({
                        "severity": key,
                        "subcategory": letter,
                        "subcategory_name": subcat,
                        "count": count,
                        "items": pieces,
                        "raw": desc,
                    })
            totals[key] = total

        matched_block = self._extract_section(text, "Совпадающие находки")
        matched: Optional[int]
        if matched_block is None:
            matched = None
        else:
            mm = re.search(r"^\s*(\d+)\s*\.", matched_block, flags=re.MULTILINE)
            matched = int(mm.group(1)) if mm else 0

        sig_total = totals["significant"]
        if sig_total is None or matched is None:
            score = float("nan")
        elif matched == 0:
            score = 0.0
        else:
            score = float(matched / (matched + sig_total))
        return ParsedGreen(score=score, sig_total=sig_total, insig_total=totals["insignificant"], matched_findings=matched, errors=errors)

    def score_pairs(self, refs: Sequence[str], cands: Sequence[str]) -> pd.DataFrame:
        prompts = [self.make_prompt(r, c) for r, c in zip(refs, cands)]
        responses: List[str] = []
        for start in range(0, len(prompts), self.batch_size):
            responses.extend(self.generate_batch(prompts[start : start + self.batch_size]))
        rows: List[Dict[str, Any]] = []
        for response in responses:
            parsed = self.parse(response)
            rows.append({
                "ru_GREEN_Qwen": parsed.score,
                "ru_GREEN_Qwen_sig_errors": parsed.sig_total,
                "ru_GREEN_Qwen_insig_errors": parsed.insig_total,
                "ru_GREEN_Qwen_matched_findings": parsed.matched_findings,
                "ru_GREEN_Qwen_errors_json": json.dumps(parsed.errors, ensure_ascii=False),
                "ru_GREEN_Qwen_analysis": response,
            })
        return pd.DataFrame(rows)


def calculate_green_for_csv(
    input_csv: str,
    output_csv: str,
    candidate_col: str = CANDIDATE_COL,
    reference_col: str = REFERENCE_COL,
    row_id_col: str = ROW_ID_COL,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 1,
    gpu: str = "0",
    device: str = "cuda",
    max_new_tokens: int = 768,
) -> str:
    configure_worker_environment(gpu=gpu, device=device)
    df = pd.read_csv(input_csv)
    if candidate_col not in df.columns or reference_col not in df.columns:
        raise ValueError(f"Input must contain '{candidate_col}' and '{reference_col}' columns")
    scorer = ManualGREENQwen(model_name=model_name, batch_size=batch_size, max_new_tokens=max_new_tokens, device=device)
    try:
        scores = scorer.score_pairs(df[reference_col].fillna("").astype(str).tolist(), df[candidate_col].fillna("").astype(str).tolist())
        out = pd.DataFrame({row_id_col: df[row_id_col].astype(str).tolist() if row_id_col in df.columns else [str(i) for i in range(len(df))]})
        out = pd.concat([out, scores], axis=1)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    finally:
        cleanup_cuda(scorer)
    return output_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calculate Qwen2.5 GREEN-like Russian report metric.")
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--candidate-col", default=CANDIDATE_COL)
    p.add_argument("--reference-col", default=REFERENCE_COL)
    p.add_argument("--row-id-col", default=ROW_ID_COL)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gpu", default="0")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=768)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    calculate_green_for_csv(**vars(args))


if __name__ == "__main__":
    main()
