"""Optional local MLflow tracking, historical imports, and dashboard startup."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from squad_qa.io import load_json

TRACKING_DIR = os.path.join("outputs", "mlflow")
EXPERIMENT = "squad2-faiss-lora"
ADAPTER_FILES = (
    "adapter_model.safetensors", "qa_head.safetensors", "adapter_config.json", "qa_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt",
)


def local_paths(directory):
    if "://" in str(directory) or str(directory).startswith(("\\\\", "//")):
        raise ValueError("Tracking requires a local directory, not a remote URI")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(exist_ok=True)
    return root, "sqlite:///" + (root / "tracking.db").as_posix()


def local_client(directory):
    if importlib.util.find_spec("mlflow") is None:
        raise RuntimeError(
            "MLflow tracking is optional. Install requirements-tracking.txt to enable it."
        )
    os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    from mlflow.tracking import MlflowClient
    root, uri = local_paths(directory)
    return root, MlflowClient(tracking_uri=uri, registry_uri=uri)


def check_local_artifacts(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc:
        raise ValueError("Tracking artifacts must use a local file URI")


def experiment_id(client, root, name):
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        suffix = hashlib.sha256(name.encode()).hexdigest()[:16]
        location = (root / "artifacts" / suffix).as_uri()
        return client.create_experiment(name, artifact_location=location)
    check_local_artifacts(experiment.artifact_location)
    return experiment.experiment_id


class LocalRun:
    def __init__(self, directory, experiment, name, tags=None, run_id=None):
        self.root, self.client = local_client(directory)
        self.experiment_id = experiment_id(self.client, self.root, experiment)
        self.name, self.tags, self.run_id = name, tags or {}, run_id

    def __enter__(self):
        if self.run_id:
            run = self.client.get_run(self.run_id)
            if run.info.experiment_id != self.experiment_id:
                raise ValueError("MLflow run belongs to a different experiment")
            check_local_artifacts(run.info.artifact_uri)
            self.client.update_run(self.run_id, status="RUNNING")
        else:
            run = self.client.create_run(
                self.experiment_id, tags={"mlflow.runName": self.name, **self.tags}
            )
            self.run_id = run.info.run_id
        print(f"MLflow run: {self.run_id}; local store: {self.root}", flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        status = "FINISHED" if exc_type is None else "FAILED"
        if exc_type is KeyboardInterrupt:
            status = "KILLED"
        self.client.set_terminated(self.run_id, status=status)
        return False

    def log_params(self, values):
        from mlflow.entities import Param
        params = [
            Param(key, json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                  else str(value))
            for key, value in values.items() if value is not None
        ]
        if params:
            self.client.log_batch(self.run_id, params=params, synchronous=True)

    def log_metrics(self, values, step=0):
        from mlflow.entities import Metric
        metrics = []
        timestamp = int(time.time() * 1000)
        for key, value in values.items():
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Non-finite MLflow metric: {key}")
            metrics.append(Metric(key, number, timestamp, step))
        if metrics:
            self.client.log_batch(self.run_id, metrics=metrics, synchronous=True)

    def set_tags(self, values):
        for key, value in values.items():
            self.client.set_tag(self.run_id, key, str(value))

    def artifact(self, path, folder="reports"):
        self.client.log_artifact(self.run_id, str(path), artifact_path=folder)


def add_tracking_arguments(parser, resume=False):
    parser.add_argument("--mlflow", action="store_true", help="Enable local MLflow tracking")
    parser.add_argument("--mlflow-dir", default=TRACKING_DIR)
    parser.add_argument("--mlflow-experiment", default=EXPERIMENT)
    parser.add_argument("--mlflow-run-name")
    parser.set_defaults(mlflow_run_id=None)
    if resume:
        parser.add_argument("--mlflow-run-id", help="Append evaluation to an existing tracking run")


def optional_run(args, default_name, stage):
    if not args.mlflow:
        if hasattr(args, "mlflow_run_id") and args.mlflow_run_id:
            raise ValueError("--mlflow-run-id requires --mlflow")
        return nullcontext(None)
    return LocalRun(
        args.mlflow_dir, args.mlflow_experiment, args.mlflow_run_name or default_name,
        tags={"stage": stage, "imported": "false"}, run_id=args.mlflow_run_id,
    )


def qa_metrics(metrics, prefix):
    result = {}
    for group, label in (("all", "overall"), ("hasans", "answerable"), ("noans", "unanswerable")):
        values = metrics[group]
        result[f"{prefix}.{label}_em"] = values["exact"]
        result[f"{prefix}.{label}_f1"] = values["f1"]
        result[f"{prefix}.{label}_count"] = values["total"]
    result[f"{prefix}.unanswerable_accuracy"] = metrics["noans"]["exact"]
    return result


def retrieval_metrics(values, prefix):
    return {f"{prefix}.{key.replace('@', '_at_')}": value for key, value in values.items()}


def log_training_config(run, history):
    keys = (
        "base_model", "seed", "train_questions", "train_features", "train_path",
        "training_context", "lr", "micro_batch", "accum", "precision",
        "max_length", "stride", "max_answer_length", "initial_epochs", "max_epochs",
        "input_sha256", "trainable_parameters", "training_mode", "encoder_mode", "lora_enabled",
    )
    params = {key: history[key] for key in keys if key in history}
    for group in ("lora", "early_stopping"):
        params.update({f"{group}.{key}": value for key, value in history.get(group, {}).items()
                       if key != "stale_epochs"})
    run.log_params(params)
    run.log_metrics(retrieval_metrics(history.get("tuning_retrieval", {}), "retrieval.tuning"))


def log_epoch(run, entry):
    metrics = qa_metrics(entry["tuning"], "tuning")
    metrics.update({"train.loss": entry["loss"], "tuning.threshold": entry["threshold"]})
    for key in ("optimizer_steps", "elapsed_seconds"):
        if key in entry:
            metrics[f"epoch.{key}"] = entry[key]
    run.log_metrics(metrics, step=entry["epoch"])


def log_evaluation(run, results, path):
    run.log_params({
        "evaluation.adapter": results["adapter"],
        "evaluation.threshold": results["config"]["threshold"],
    })
    for condition in ("oracle", "retrieved"):
        run.log_metrics(qa_metrics(results[condition], f"dev.{condition}"))
    run.log_metrics(retrieval_metrics(results["retrieval"], "retrieval.dev"))
    run.artifact(path)


def log_training_result(run, history, selection_path, log_adapter=False):
    best = next(entry for entry in history["epochs"]
                if entry["epoch"] == history["selected_epoch"])
    run.log_metrics(qa_metrics(best["tuning"], "selected.tuning"))
    run.log_metrics({
        "selection.epoch": best["epoch"], "selection.threshold": best["threshold"],
        "training.completed_epochs": len(history["epochs"]),
        "training.wall_seconds": history.get("elapsed_seconds"),
    })
    run.set_tags({
        "selected_adapter": history["selected_adapter"],
        "stop_reason": history.get("stop_reason", "not recorded"),
    })
    run.artifact(selection_path)
    adapter = Path(history["selected_adapter"])
    config = adapter / "qa_config.json"
    if config.is_file():
        run.artifact(config, "selected_adapter")
    evaluation = adapter / "dev_results.json"
    if evaluation.is_file():
        log_evaluation(run, load_json(evaluation), evaluation)
    diagnostic = adapter / "tuning_diagnostics" / "summary.json"
    if diagnostic.is_file():
        report = load_json(diagnostic)
        for condition, summary in report["conditions"].items():
            answerable = summary["answerable"]
            run.log_metrics({
                f"diagnostic.{condition}.answerable_f1_before": answerable["before_abstention"]["f1"],
                f"diagnostic.{condition}.answerable_f1_after": answerable["after_abstention"]["f1"],
                **{f"diagnostic.{condition}.{key}": value
                   for key, value in answerable["buckets"].items()},
            })
        run.artifact(diagnostic, "diagnostics")
    if log_adapter:
        for name in ADAPTER_FILES:
            path = adapter / name
            if path.is_file():
                run.artifact(path, "selected_adapter")


def log_external_result(run, results, path):
    run.log_params(results["config"])
    run.set_tags({"primary_decision": "zero_margin", "training": "external checkpoint"})
    calibration = results["calibration"]
    run.log_metrics(qa_metrics(calibration["zero_margin_metrics"], "tuning.zero_margin"))
    run.log_metrics(qa_metrics(calibration["calibrated_metrics"], "tuning.calibrated"))
    run.log_metrics({"calibration.threshold": calibration["threshold"],
                     "evaluation.wall_seconds": results["elapsed_seconds"]})
    for mode in ("zero_margin", "calibrated"):
        for condition in ("oracle", "retrieved"):
            metrics = results["dev"][mode][condition]
            run.log_metrics(qa_metrics(metrics, f"dev.{mode}.{condition}"))
            if mode == "zero_margin":
                run.log_metrics(qa_metrics(metrics, f"dev.{condition}"))
    for split, values in results["retrieval"].items():
        run.log_metrics(retrieval_metrics(values, f"retrieval.{split}"))
    run.artifact(path)


def import_run(source, directory, experiment, log_adapter=False):
    source = Path(source).resolve()
    document = load_json(source)
    training = "epochs" in document and "selected_epoch" in document
    external = "dev" in document and "calibration" in document and "config" in document
    if not training and not external:
        raise ValueError(f"Expected a training selection or external-reader results JSON: {source}")
    files = [source]
    if training:
        adapter = Path(document["selected_adapter"])
        files += [p for p in (adapter / "dev_results.json",
                             adapter / "tuning_diagnostics" / "summary.json") if p.is_file()]
    fingerprint = hashlib.sha256()
    for path in files:
        fingerprint.update(str(path.resolve()).encode())
        fingerprint.update(path.read_bytes())
    fingerprint.update(str(log_adapter).encode())
    value = fingerprint.hexdigest()
    root, client = local_client(directory)
    experiment_key = experiment_id(client, root, experiment)
    matches = client.search_runs(
        [experiment_key], filter_string=f"tags.`import.fingerprint` = '{value}'", max_results=10
    )
    for match in matches:
        if match.info.status == "FINISHED":
            print(f"Already imported: {source.name}; run={match.info.run_id}", flush=True)
            return match.info.run_id
    with LocalRun(directory, experiment, source.parent.name + " (imported)", tags={
        "imported": "true", "import.fingerprint": value, "import.source": str(source),
        "stage": "historical training" if training else "historical external evaluation",
    }) as run:
        if training:
            log_training_config(run, document)
            for entry in document["epochs"]:
                log_epoch(run, entry)
            log_training_result(run, document, source, log_adapter)
        else:
            log_external_result(run, document, source)
        status_path = source.parent / "run_status.json"
        if status_path.is_file():
            status = load_json(status_path)
            run.set_tags({f"original.{key}": status[key]
                          for key in ("started_at", "completed_at") if key in status})
        return run.run_id


def serve_ui(directory, port):
    from importlib.metadata import PackageNotFoundError, version
    try:
        version("mlflow")
    except PackageNotFoundError as exc:
        raise RuntimeError("Start the dashboard with .venv-mlflow, not the training environment.") from exc
    root, uri = local_paths(directory)
    os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    from mlflow.cli import cli
    print(f"Local dashboard: http://127.0.0.1:{port}", flush=True)
    cli.main(args=[
        "server", "--backend-store-uri", uri,
        "--default-artifact-root", (root / "artifacts").as_uri(),
        "--host", "127.0.0.1", "--port", str(port), "--workers", "1",
    ], prog_name="mlflow")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importing = commands.add_parser("import")
    importing.add_argument("--source", nargs="+", required=True)
    importing.add_argument("--experiment", default=EXPERIMENT)
    importing.add_argument("--log-adapter", action="store_true")
    dashboard = commands.add_parser("ui")
    dashboard.add_argument("--port", type=int, default=5000)
    for command in (importing, dashboard):
        command.add_argument("--tracking-dir", default=TRACKING_DIR)
    args = parser.parse_args()
    if args.command == "import":
        for source in args.source:
            import_run(source, args.tracking_dir, args.experiment, args.log_adapter)
    else:
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        serve_ui(args.tracking_dir, args.port)


if __name__ == "__main__":
    main()
