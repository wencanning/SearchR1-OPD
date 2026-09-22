"""Reject stale experiment code or changed checkpoint files before each stage."""
import json
from common import ROOT,OUT,sha,verify_inputs

m=verify_inputs()
frozen=json.loads((OUT/'execution.frozen.json').read_text())
assert sha(OUT/'inputs.frozen.json')==frozen['input_sha256'],'Frozen inputs changed'
for path,h in frozen['code_sha256'].items():assert sha(ROOT/path)==h,('Frozen code changed',path)
for path,snapshot in frozen['checkpoint_files'].items():
    stat=(ROOT/path).stat();assert stat.st_size==snapshot['size'] and stat.st_mtime_ns==snapshot['mtime_ns'],('Checkpoint file changed',path)
assert json.loads((OUT/'ENV_KERNEL_WITNESS.json').read_text())['passed']
print('EXECUTION_GUARD_PASS',flush=True)
