"""Tracking stays optional; SQLite integration runs only with the tracking extra installed."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from squad_qa import mlflow_tracking as tracking


class TrackingTests(unittest.TestCase):
    def test_disabled_tracking_does_not_create_a_client(self):
        with patch.object(tracking, "LocalRun") as run:
            with tracking.optional_run(argparse.Namespace(mlflow=False), "test", "training") as value:
                self.assertIsNone(value)
            run.assert_not_called()

    def test_run_id_requires_tracking(self):
        with self.assertRaisesRegex(ValueError, "requires --mlflow"):
            tracking.optional_run(
                argparse.Namespace(mlflow=False, mlflow_run_id="existing"), "test", "evaluation"
            )

    def test_remote_tracking_and_artifacts_rejected(self):
        with self.assertRaisesRegex(ValueError, "local directory"):
            tracking.local_paths("https://example.com/tracking")
        with self.assertRaisesRegex(ValueError, "local directory"):
            tracking.local_paths(r"\\server\share")
        for uri in ("s3://bucket/model", "https://example.com/artifacts", "file://server/share"):
            with self.subTest(uri=uri), self.assertRaisesRegex(ValueError, "local file"):
                tracking.check_local_artifacts(uri)
        tracking.check_local_artifacts("file:///C:/project/artifacts")

    def test_metric_names_preserve_subgroup_counts_and_undefined_values(self):
        source = {
            "all": {"exact": 70, "f1": 75, "total": 2},
            "hasans": {"exact": 60, "f1": 70, "total": 1},
            "noans": {"exact": 80, "f1": 80, "total": 1},
        }
        values = tracking.qa_metrics(source, "tuning")
        self.assertEqual(values["tuning.overall_f1"], 75)
        self.assertEqual(values["tuning.answerable_f1"], 70)
        self.assertEqual(values["tuning.unanswerable_accuracy"], 80)
        self.assertEqual(values["tuning.answerable_count"], 1)
        self.assertEqual(
            tracking.retrieval_metrics({"recall@3_all": 0.85}, "retrieval.dev"),
            {"retrieval.dev.recall_at_3_all": 0.85},
        )

    @unittest.skipUnless(importlib.util.find_spec("mlflow"), "Install requirements-tracking.txt")
    def test_local_sqlite_lifecycle_history_import_and_artifacts(self):
        # A child process releases MLflow's pooled SQLite handles before Windows temp cleanup.
        script = r'''
import json
from pathlib import Path
import sys
from squad_qa.io import save_json
from squad_qa.mlflow_tracking import (
    LocalRun, import_run, local_client, log_epoch, log_evaluation,
    log_training_config, log_training_result,
)
root=Path(sys.argv[1])
store=root/'store'
adapter=root/'epoch2'
adapter.mkdir()
def metrics(f1):
    return {'all':{'exact':f1,'f1':f1,'total':2},
            'hasans':{'exact':f1-10,'f1':f1-5,'total':1},
            'noans':{'exact':f1+5,'f1':f1+5,'total':1}}
history={
    'base_model':'fixture','train_questions':20,'seed':42,'lr':0.0002,
    'lora':{'r':16,'alpha':32,'dropout':0.05},
    'input_sha256':{'fixture.json':'abc'},
    'epochs':[{'epoch':i,'loss':1/i,'threshold':float(i),
               'tuning':metrics(score)} for i,score in [(1,50),(2,70),(3,60)]],
    'selected_epoch':2,'selected_adapter':str(adapter),'stop_reason':'early_stopping',
    'elapsed_seconds':12.5,
}
selection=root/'selection.json'
save_json(selection,history)
save_json(adapter/'qa_config.json',{'threshold':2.0})
(adapter/'adapter_model.safetensors').write_bytes(b'fixture adapter artifact')
evaluation={
    'adapter':str(adapter),'config':{'threshold':2.0},'retrieval':{'recall@3_all':0.85},
    'retrieved':metrics(74),'oracle':metrics(79),
}
save_json(adapter/'dev_results.json',evaluation)
with LocalRun(store,'tests','live',tags={'imported':'false'}) as live:
    log_training_config(live,history)
    for entry in history['epochs']:
        log_epoch(live,entry)
    log_training_result(live,history,selection,log_adapter=True)
    live_id=live.run_id
_,client=local_client(store)
assert client.get_run(live_id).info.status=='FINISHED'
assert client.get_run(live_id).data.metrics['selected.tuning.overall_f1']==70
assert client.get_run(live_id).data.metrics['tuning.overall_f1']==60
assert client.get_run(live_id).data.metrics['dev.retrieved.overall_f1']==74
assert client.get_run(live_id).data.params['lora.r']=='16'
assert len(client.get_metric_history(live_id,'tuning.overall_f1'))==3
assert {x.path for x in client.list_artifacts(live_id,'selected_adapter')}>={
    'selected_adapter/adapter_model.safetensors','selected_adapter/qa_config.json'
}
with LocalRun(store,'tests','evaluation',run_id=live_id) as resumed:
    log_evaluation(resumed,evaluation,adapter/'dev_results.json')
try:
    with LocalRun(store,'tests','failure') as failed:
        failed_id=failed.run_id
        failed.log_metrics({'bad':float('nan')})
except ValueError:
    pass
else:
    raise AssertionError('Non-finite metric accepted')
assert client.get_run(failed_id).info.status=='FAILED'
before=selection.read_bytes()
imported=import_run(selection,store,'tests')
assert imported==import_run(selection,store,'tests')
record=client.get_run(imported)
assert record.data.tags['imported']=='true'
assert record.data.metrics['selected.tuning.overall_f1']==70
assert record.data.metrics['dev.oracle.overall_f1']==79
assert selection.read_bytes()==before
external={
    'config':{'reader_model':'external-fixture','trained_in_this_run':False},
    'calibration':{'threshold':-3.0,'zero_margin_metrics':metrics(65),
                   'calibrated_metrics':metrics(68)},
    'retrieval':{'dev':{'recall@3_all':0.85}},'elapsed_seconds':4,
    'dev':{mode:{'oracle':metrics(80),'retrieved':metrics(73)}
           for mode in ('zero_margin','calibrated')},
}
source=root/'external.json'
save_json(source,external)
external_id=import_run(source,store,'tests')
assert client.get_run(external_id).data.metrics['dev.retrieved.overall_f1']==73
assert client.get_run(external_id).data.metrics['dev.calibrated.oracle.overall_f1']==80
assert client.get_run(external_id).data.tags['primary_decision']=='zero_margin'
print('VERIFIED='+json.dumps({'epochs':3,'selected_f1':70,'import_deduplicated':True}))
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", script, directory],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, encoding="utf-8",
                env=dict(os.environ, PYTHONIOENCODING="utf-8"),
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            verified = next(line for line in result.stdout.splitlines() if line.startswith("VERIFIED="))
            self.assertTrue(json.loads(verified.split("=", 1)[1])["import_deduplicated"])


if __name__ == "__main__":
    unittest.main()
