"""Freeze the next generation check without inspecting any generated outputs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/er_generation_cpu_20260916'
OLD=ROOT/'reports/action_calibration_cpu_20260916'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    m=json.loads((OLD/'inputs.frozen.json').read_text())
    prior=json.loads((OLD/'update_probe_before.json').read_text())
    used={c['qid'] for c in prior}
    available=set(m['split_qids']['test'])-used
    order=sorted(available,key=lambda q:hashlib.sha256(('generation-transfer-20260916|'+q).encode()).hexdigest())
    assert len(order)==24 and not set(order)&set(m['split_qids']['update'])
    variants=['opd','full_er','strict_body_er']
    inputs=[OLD/'inputs.frozen.json',OLD/'update_samples.jsonl',OLD/'update_targets.jsonl',
        OLD/'update_samples.frozen.json',OLD/'update_targets.frozen.json',
        OLD/'update_effects.jsonl',ROOT/'reports/strict_role_cpu_20260916/update_effects.jsonl']
    sources=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))+[
        ROOT/'scripts/experiments/action_calibration_cpu/common.py',
        ROOT/'scripts/experiments/strict_role_cpu/local_update.py',
        ROOT/'scripts/experiments/evidence_use/experiment.py',ROOT/'verl/trainer/ppo/core_algos.py']
    manifest=dict(variants=variants,pilot_qids=order[:4],full_qids=order[4:],
        excluded_old_probe_qids=sorted(used),update_qids=m['split_qids']['update'],
        selection='Hash-order remaining test questions; prior eight reference-body probes excluded. Existing action probabilities were previously inspected.',
        interpretation='Exploratory one-update generation; neither fresh full-task evaluation nor matched-seed training.',
        modes=[['both_hops','forced_answer'],['both_hops','free_action'],['with_distractors','free_action']],
        max_new_tokens=128,device='cpu',dtype='float32',threads=4,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources+inputs})
    target=OUT/'protocol.frozen.json'
    if target.exists():assert json.loads(target.read_text())==manifest,'Frozen protocol differs'
    else:target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(pilot_questions=4,manual_full_questions=20,pilot_generations=36,
        pilot_qids=order[:4],no_teacher_calls=True,no_retrieval=True,no_gpu=True),ensure_ascii=False))


if __name__=='__main__':main()
