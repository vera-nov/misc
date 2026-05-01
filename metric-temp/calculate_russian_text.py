from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
    require_column,
)

PROMPT_WORD_LIMIT = 300
MAX_MODEL_LENGTH = 2048
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def safe_apply_qwen_chat_template(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "\n".join(m["content"] for m in messages) + "\n"


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


@dataclass
class ParsedGreen:
    score: float
    sig_total: int
    insig_total: int
    matched_findings: int
    errors: Dict[str, Any]


class ManualGREENQwen:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = get_input_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model_kwargs = {
            "torch_dtype": pick_dtype(),
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device != "cpu" and torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if self.device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    def make_prompt(self, reference_text: str, candidate_text: str) -> str:
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

Клинически незначимых ошибок нет."""

    def clean_response(self, text: str) -> str:
        return strip_thinking(text).strip()

    def _tokenize_batch(self, prompts: Sequence[str]):
        messages = [[{"role": "user", "content": p}] for p in prompts]
        rendered = [safe_apply_qwen_chat_template(self.tokenizer, m) for m in messages]
        return self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_MODEL_LENGTH,
        )

    def generate_batch(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        import torch

        batch = self._tokenize_batch(prompts)
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        input_len = batch["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return [self.clean_response(self.tokenizer.decode(o[input_len:], skip_special_tokens=True)) for o in output]

    def _extract_section(self, text: str, title: str) -> str:
        pattern = rf"\[{re.escape(title)}\]\s*:\s*(.*?)(?=\n\s*\[[^\]]+\]\s*:|\Z)"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _parse_count_and_desc(self, line: str) -> tuple[int, str]:
        match = re.search(r":\s*(\d+)", line)
        count = int(match.group(1)) if match else 0
        desc = ""
        if match:
            desc = line[match.end() :].strip(" .;\t")
        return count, desc

    def parse(self, text: str) -> ParsedGreen:
        sig_section = self._extract_section(text, "Клинически значимые ошибки")
        insig_section = self._extract_section(text, "Клинически незначимые ошибки")
        matched_section = self._extract_section(text, "Совпадающие находки")
        errors: Dict[str, Any] = {"significant": {}, "insignificant": {}}
        sig_total = 0
        insig_total = 0
        for label, section in (("significant", sig_section), ("insignificant", insig_section)):
            for line in section.splitlines():
                if not line.strip().startswith("("):
                    continue
                count, desc = self._parse_count_and_desc(line)
                errors[label][line.strip()[:3]] = {"count": count, "description": desc}
                if label == "significant":
                    sig_total += count
                else:
                    insig_total += count
        match = re.search(r"(\d+)", matched_section)
        matched = int(match.group(1)) if match else 0
        if not matched:
            score = 0
        else:
            score = matched / (matched + sig_total)
        return ParsedGreen(score=score, sig_total=sig_total, insig_total=insig_total, matched_findings=matched, errors=errors)

    def score_pairs(
        self,
        references: Sequence[str],
        candidates: Sequence[str],
        batch_size: int = 1,
        max_new_tokens: int = 512,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        prompts = [self.make_prompt(str(r), str(c)) for r, c in zip(references, candidates)]
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            outputs = self.generate_batch(batch_prompts, max_new_tokens=max_new_tokens)
            for output in outputs:
                try:
                    parsed = self.parse(output)
                except Exception as exc:
                    warnings.warn(f"Could not parse GREEN output: {exc}")
                    parsed = ParsedGreen(score=np.nan, sig_total=0, insig_total=0, matched_findings=0, errors={})
                rows.append(
                    {
                        "ru_GREEN_Qwen": parsed.score,
                        "ru_GREEN_Qwen_sig_errors": parsed.sig_total,
                        "ru_GREEN_Qwen_insig_errors": parsed.insig_total,
                        "ru_GREEN_Qwen_matched_findings": parsed.matched_findings,
                        "ru_GREEN_Qwen_errors_json": json.dumps(parsed.errors, ensure_ascii=False),
                        "ru_GREEN_Qwen_analysis": output,
                    }
                )
        return pd.DataFrame(rows)


def calculate_green_for_csv(
    input_csv: str,
    output_csv: str,
    model_name: str = DEFAULT_MODEL,
    candidate_col: str = CANDIDATE_COL,
    reference_col: str = REFERENCE_COL,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    device: Optional[str] = None,
    gpu: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    configure_worker_environment(gpu)
    df = pd.read_csv(input_csv)
    for col in (ROW_ID_COL, candidate_col, reference_col):
        require_column(df, col)
    scorer = ManualGREENQwen(model_name=model_name, device=device)
    scores = scorer.score_pairs(
        references=df[reference_col].fillna("").astype(str).tolist(),
        candidates=df[candidate_col].fillna("").astype(str).tolist(),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    out = pd.concat([df[[ROW_ID_COL]].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    out.to_csv(output_csv, index=False)
    cleanup_cuda(scorer)
    return out
