# Radiology report quality pipeline

Скрипты:

- `build_method.py` — считает все базовые метрики, делает nested CV, выбирает финальную комбинацию и сохраняет JSON-конфиг.
- `evaluate.py` — читает JSON-конфиг, считает только нужные метрики и сохраняет оценки по строкам + mean/std.
- `calculate_russian_text.py` — GREEN-like оценка на русском через `Qwen/Qwen2.5-7B-Instruct`.
- `translate_reports.py` — перевод RU→EN через `qwen`, `hy_mt`, `translategemma`.
- `calculate_english_text.py` — RadEval, cosine и BERTScore-like метрики на английских текстах.
- `calculate_image_text.py` — только BioViL-T image-text cosine.
- `pipeline_common.py` — общие функции, реестр методов, nested CV, multiprocessing worker.

## Ожидаемый CSV

Минимум 4 колонки: кандидат, эталон, путь к CXR, врачебная оценка. Скрипты пытаются определить названия автоматически, но надежнее передать их явно.

## build

```bash
cd radiology_quality_pipeline
uv run python build_method.py /path/to/train.csv \
  --candidate-col "отчет-кандидат" \
  --reference-col "отчет-эталон" \
  --image-col "путь к изображению" \
  --target-col "врачебная оценка качества отчета" \
  --config-out final_config.json \
  --gpu 0 \
  --device cuda \
  --k-outer 5 \
  --k-inner 5 \
  --weight-step 0.05 \
  --target-direction higher_is_better \
  --verbose
```

Если врачебная оценка устроена как число ошибок, где меньше — лучше, используйте:

```bash
--target-direction lower_is_better
```

## evaluate

```bash
cd radiology_quality_pipeline
uv run python evaluate.py \
  --config final_config.json \
  --path_to_data_description_for_evaluation /path/to/eval.csv \
  --output eval_with_scores.csv \
  --gpu 0 \
  --device cuda \
  --verbose
```

В CSV будут добавлены выбранные feature-колонки, `quality_score`, `quality_score_mean`, `quality_score_std`, а если в ансамбль попал GREEN — `llm_errors_json`.
