# Extractive Question Answering with FAISS and LoRA

A SQuAD 2.0 question-answering pipeline that retrieves Wikipedia paragraphs,
extracts an answer span, and abstains when the evidence is insufficient.

- **Retriever:** MiniLM embeddings and FAISS find relevant paragraphs.
- **Reader:** RoBERTa predicts the answer's start and end positions.
- **Fine-tuning:** LoRA adapts the encoder while a span head is trained normally.

The 20k-example LoRA model reached **74.0 F1 and 71.5 exact match** on a balanced
200-question dev subset. Training updated 591,362 parameters, about **0.47%**
of the adapted model, on an RTX 3050 Ti GPU with 4 GB VRAM.

## Results

All models below use the same top-3 retrieval pipeline and 200 dev questions:
100 answerable and 100 unanswerable. EM is normalized exact match; F1 gives
partial credit for answer-word overlap. The official scorer takes the best
match across all reference answers.

| Model | Answerable F1 | Unanswerable accuracy | Overall EM / F1 |
|---|---:|---:|---:|
| RoBERTa + LoRA, 3k examples | 27.3 | 86% | 56.0 / 56.7 |
| **RoBERTa + LoRA, 20k examples** | **62.0** | **86%** | **71.5 / 74.0** |
| External SQuAD2 reader, zero margin | 72.7 | 73% | 69.0 / 72.8 |
| Frozen RoBERTa + trained span head, 20k | 0.0 | 96% | 48.0 / 48.0 |

With the gold paragraph supplied instead of retrieved context, overall F1 is
78.8 for the 20k LoRA model and 80.3 for the external reader.

The expanded LoRA run gained 17.4 overall F1 points over the 3k run. The external
reader answers answerable questions better, while the LoRA model rejects more
unanswerable ones.

These are development results, not full SQuAD2 benchmark scores. Full validation
contains 11,873 questions and has not yet been evaluated. The external model,
[`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2),
was already fine-tuned on SQuAD2; it is a quality reference, not a controlled
training-cost comparison.

## How it works

```text
Question -> MiniLM -> FAISS -> top-3 distinct paragraphs
                                       |
                         question + paragraph windows
                                       |
                           RoBERTa + LoRA -> span head
                                       |
                             answer or abstention
```

Paragraphs are indexed as overlapping chunks, with each hit mapped back to
its parent paragraph. The reader scores windows from the retrieved paragraphs.
It returns a span from the original text or abstains using a threshold chosen
on tuning data.

| Component | Setting |
|---|---|
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` |
| Search | FAISS `IndexFlatIP` over normalized embeddings |
| Embedding chunks | 254 tokens, advancing 127 tokens per chunk |
| Reader | `FacebookAI/roberta-base` with 384-token windows and 128-token context overlap |
| LoRA | Rank 16, alpha 32, dropout 0.05; attention `query` and `value` projections |
| Span output | At most 30 tokens from one window |
| Training | Learning rate 2e-4; effective batch size 16; seed 42 |

The original encoder weights remain frozen. The LoRA adapters contribute
589,824 trainable parameters and the span head adds 1,538. The frozen backbone
still consumes memory and compute; fewer trainable parameters do not imply an
equivalent reduction in total resource use.

## Data and evaluation

Source: [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2), with
130,319 training questions and 11,873 public validation questions.

| Project split | Size | Purpose |
|---|---:|---|
| Training | 3k initially, then 20k | Update adapters and span head |
| Tuning | 300 | Select checkpoints, stopping point, and abstention threshold |
| Dev | 200 | Compare the selected models |

Each subset is balanced between answerable and unanswerable questions. Thirty
training article titles are reserved for tuning; the other 412 supply training
examples. Neither group includes validation titles. Expansion retains the
original 3k examples and preserves the tuning/dev files, checked by file hashes.

Training uses gold paragraphs. Tuning and dev evaluation use separate FAISS
indexes covering their respective document pools. Checkpoint and threshold
selection use tuning only; the saved threshold is applied unchanged on dev.
Metrics come from the [official SQuAD2 scorer](squad_qa/squad2_metrics.py).

The 200 dev questions informed development choices, so they are not an untouched
test set. The external reader may also have seen the project's tuning questions
during its original SQuAD2 training.

## Experiments

- **Chunking long paragraphs.** On 1,081 validation questions with long gold
  paragraphs, chunking raised answerable recall@3 from 71.2% to 81.7%. This was
  a retrieval-only study. Answerable recall@3 on the fixed dev subset is 85%.
- **More data and adaptive training.** Expanding from 3k to 20k examples and
  allowing a longer run raised retrieved F1 from 56.7 to 74.0. Training stopped
  at epoch 9 and retained epoch 6, the tuning-best checkpoint. The experiment
  changed both data size and training duration; it does not isolate either.
- **Head-only ablation.** A frozen encoder with only the span head trained
  reached 48.0 dev F1 and mostly abstained. This supports encoder adaptation
  under the tested recipe. 
- **Error analysis.** On 150 answerable tuning questions, 7 exactly correct
  spans were suppressed by abstention. The gold paragraph was missing from
  retrieval for 42 questions. Candidate F1 was 77.8 when the gold paragraph was
  present versus 14.6 when absent, making retrieval a clear area for further work.

## Run the project

Tested with Python 3.12 on Windows and an NVIDIA CUDA GPU. The requirements
include CUDA 12.4 PyTorch wheels. Reader commands support CPU, but the eval-index
builder below uses CUDA. Initial dataset/model downloads require internet access.

### Setup and data

Run from the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

.\.venv\Scripts\python.exe -m squad_qa.data_prep
.\.venv\Scripts\python.exe -m squad_qa.prepare_lora_data
.\.venv\Scripts\python.exe -m squad_qa.prepare_lora_data --expand-train-size 20000
.\.venv\Scripts\python.exe -m squad_qa.build_index --variant flat
.\.venv\Scripts\python.exe -m squad_qa.build_index --variant chunked
.\.venv\Scripts\python.exe -m squad_qa.build_training_index
```

For existing experiments, reuse the prepared files rather than regenerating
splits. Build the flat index before the chunked one. Data, indexes, checkpoints,
and results are generated under the git-ignored `data/` and `outputs/` folders.
Trained checkpoints are not bundled with the repository.

### Evaluate the external reader without training

```powershell
.\.venv\Scripts\python.exe -m squad_qa.external_reader --batch-size 2 --output-dir outputs\external_baseline
```

This reports both zero-margin and tuning-calibrated results without updating
weights. The results table uses zero margin under this project's decoder,
not Hugging Face pipeline defaults.

### Train and evaluate RoBERTa + LoRA

```powershell
.\.venv\Scripts\python.exe -m squad_qa.run_roberta train --train-path data\train_20000.json --epochs 5 --max-epochs 15 --patience 3 --min-delta 0.1 --micro-batch 1 --accum 16 --batch-size 2 --output-dir outputs\roberta_20k

$run = Get-Content outputs\roberta_20k\selection.json -Raw | ConvertFrom-Json
$adapter = $run.selected_adapter

.\.venv\Scripts\python.exe -m squad_qa.run_roberta evaluate --adapter $adapter --batch-size 2
.\.venv\Scripts\python.exe -m squad_qa.run_roberta ask --adapter $adapter --question "Who built the Eiffel Tower?" --batch-size 2
.\.venv\Scripts\python.exe -m squad_qa.diagnose_reader --adapter $adapter --batch-size 2
```

The run trains for at least 5 epochs and stops after 3 epochs without a
0.1-point tuning F1 improvement, up to 15 epochs. `selection.json` identifies
the best checkpoint. Use a fresh output directory for each experiment.

Add `--training-mode head_only` to train the frozen-encoder baseline. Evaluation
and diagnostics load either checkpoint type automatically. `evaluate` currently
scores dev_200; diagnostics use tuning only. Exact optimizer-state resume is
not yet supported.

## Optional MLflow tracking

Add `--mlflow` to training or evaluation to record parameters, epoch curves,
selected-checkpoint metrics, and reports. Tracking is local and off by default.
The dashboard uses a separate environment to avoid changing training dependencies.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tracking.txt
.\.venv\Scripts\python.exe -m venv .venv-mlflow
.\.venv-mlflow\Scripts\python.exe -m pip install -r requirements-mlflow-ui.txt
.\.venv-mlflow\Scripts\python.exe -m squad_qa.mlflow_tracking ui --port 5000
```

Open `http://127.0.0.1:5000` and select the `squad2-faiss-lora` experiment.
Logging does not require the dashboard to be running. Use
`selected.tuning.overall_f1` to compare chosen checkpoints; `tuning.overall_f1`
shows the per-epoch curve.

Evaluation can append to a tracked training run with
`--mlflow --mlflow-run-id $run.mlflow_run_id`. Completed, previously untracked
experiments can be imported with `python -m squad_qa.mlflow_tracking import --source`
followed by their selection or external-result JSON paths. Imports are labeled
as historical and unchanged imports are skipped.

## Code

- [retrieval.py](squad_qa/retrieval.py): chunking, embeddings, and FAISS search.
- [roberta_qa.py](squad_qa/roberta_qa.py): span features, checkpoint loading, and decoding.
- [run_roberta.py](squad_qa/run_roberta.py): training, stopping, evaluation, and inference.
- [external_reader.py](squad_qa/external_reader.py): off-the-shelf QA baseline.
- [diagnose_reader.py](squad_qa/diagnose_reader.py): extraction versus abstention errors.
- [mlflow_tracking.py](squad_qa/mlflow_tracking.py): optional MLflow logging and historical imports.
- [tests/](tests/): split integrity, span alignment, checkpoint, and tracking tests.
