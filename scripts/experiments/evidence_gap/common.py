"""CPU-only protocol, split, command construction and summary helpers."""
import json
import math
import re
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name('protocol.json')
METHODS = ('opd', 'sod', 'er')
TARGETS = ('hidden', 'observed', 'er', 'temperature')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n'); tmp.replace(path)


def frozen(path, obj):
    if Path(path).exists():
        if read(path) != obj:
            raise ValueError(f'Existing input differs: {path}; use a separate output directory for a new protocol')
    else:
        save(path, obj)


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def append(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n'); f.flush()


def config(path=DEFAULT_CONFIG):
    c = read(path)
    for k in ('output', 'training_python', 'probe_python', 'student', 'teacher', 'train_file',
              'test_file', 'old_validation_file', 'legacy_checkpoint_root'):
        # Preserve the venv interpreter symlink: realpath would select the base
        # Python executable and silently lose the environment's packages.
        p=ROOT/c[k]
        c[k] = str(p.absolute() if k.endswith('_python') else p.resolve())
    return c


def normalize(s):
    s = ''.join(x for x in (s or '').lower() if x not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def exact(answer, gold):
    return answer is not None and any(normalize(answer) == normalize(g) for g in gold)


def f1(answer, gold):
    from collections import Counter
    a = normalize(answer).split()
    def one(g):
        b = normalize(g).split()
        if not a or not b: return float(a == b)
        common = sum((Counter(a) & Counter(b)).values())
        return 2 * common / (len(a) + len(b))
    return max(map(one, gold), default=0.0)


def run_dir(c, method, seed):
    return Path(c['output']) / 'training' / f'{method}-seed{seed}'


def checkpoint(c, method, seed, step):
    if step == 0: return c['student']
    base = Path(c['legacy_checkpoint_root']) if method == 'legacy' else run_dir(c, method, seed) / 'actor'
    p = base / f'global_step_{step}'
    if not (p / 'config.json').exists(): raise FileNotFoundError(f'Missing planned checkpoint: {p}')
    return str(p)


def train_command(c, method, seed, smoke=False):
    """Same GRPO, rollout and optimizer budget; only target/SOD weighting differs."""
    out = run_dir(c, method, seed) if not smoke else Path(c['output']) / 'smoke' / method
    opts = {
      'data.train_files': c['train_file'], 'data.val_files': str(Path(c['output'])/'development.parquet'),
      'data.train_data_source': 'hotpotqa', 'data.val_data_source': 'hotpotqa',
      'data.train_seed': seed, 'data.train_batch_size': c['train_batch_size'], 'data.val_batch_size': 128,
      'data.max_prompt_length': c['max_context'], 'data.max_response_length': c['max_action_tokens'],
      'data.max_start_length': 2048, 'data.max_obs_length': c['max_observation_tokens'],
      'algorithm.adv_estimator': 'opd', 'algorithm.opd.advantage_mode': 'token',
      'algorithm.opd.normalize': False, 'algorithm.opd.clip_value': None,
      'algorithm.opd.distillation_coef': 1.0, 'algorithm.opd.lambda_distill': c['lambda_distill'],
      'algorithm.opd.teacher_target': 'evidence_residual' if method == 'er' else 'observed',
      'algorithm.opd.evidence_residual_alpha': c['alpha'], 'algorithm.opd.grpo_reward_coef': 1.0,
      'algorithm.opd.mask_protocol_tags': False, 'algorithm.opd.use_gated_distillation': False,
      'algorithm.opd.rce.enable': False, 'algorithm.opd.sod.enable': method == 'sod',
      'algorithm.opd.sod.allow_no_grpo_ablation': False, 'algorithm.opd.sod.epsilon': 1e-6,
      'algorithm.opd.sod.delta': 0.2, 'algorithm.opd.target_token_chunk_size': 256,
      'algorithm.no_think_rl': False, 'algorithm.opd.diagnostics.enable': False,
      'actor_rollout_ref.model.path': c['student'], 'actor_rollout_ref.ref.model_path': c['teacher'],
      'actor_rollout_ref.model.enable_gradient_checkpointing': True,
      'actor_rollout_ref.model.use_remove_padding': True,
      'actor_rollout_ref.ref.attn_implementation': 'sdpa',
      'actor_rollout_ref.ref.observed_target_backend': 'padded',
      'actor_rollout_ref.ref.target_forward_dtype': 'float16',
      'actor_rollout_ref.ref.allow_approximate_target_precision': True,
      'actor_rollout_ref.ref.target_select_policy_logits': True,
      'actor_rollout_ref.ref.target_trim_shared_prompt_padding': True,
      'actor_rollout_ref.ref.allow_tf32': False,
      'actor_rollout_ref.ref.fsdp_config.model_dtype': 'float16',
      'actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype': 'float16',
      'actor_rollout_ref.ref.log_prob_micro_batch_size': 2,
      'actor_rollout_ref.ref.log_prob_use_dynamic_bsz': False,
      'actor_rollout_ref.ref.fsdp_config.param_offload': False,
      'actor_rollout_ref.actor.optim.lr': c['learning_rate'],
      'actor_rollout_ref.actor.ppo_mini_batch_size': c['train_batch_size'],
      'actor_rollout_ref.actor.ppo_micro_batch_size': 2,
      'actor_rollout_ref.actor.ppo_epochs': 1,
      'actor_rollout_ref.actor.clip_ratio_low': 0.2, 'actor_rollout_ref.actor.clip_ratio_high': 0.28,
      'actor_rollout_ref.actor.clip_ratio_c': 3.0, 'actor_rollout_ref.actor.use_kl_loss': False,
      'actor_rollout_ref.actor.entropy_coeff': 0.0, 'actor_rollout_ref.actor.state_masking': True,
      'actor_rollout_ref.actor.fsdp_config.param_offload': False,
      'actor_rollout_ref.actor.fsdp_config.optimizer_offload': False,
      'actor_rollout_ref.rollout.log_prob_micro_batch_size': 4,
      'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
      'actor_rollout_ref.rollout.gpu_memory_utilization': 0.35,
      'actor_rollout_ref.rollout.free_cache_engine': False, 'actor_rollout_ref.rollout.enforce_eager': True,
      'actor_rollout_ref.rollout.seed': seed, 'actor_rollout_ref.rollout.restrict_to_tokenizer_vocab': True,
      'actor_rollout_ref.rollout.n': 1, 'actor_rollout_ref.rollout.n_agent': c['n_agent'],
      'trainer.logger': '[console]', '+trainer.val_before_train': False, '+trainer.val_only': False,
      'trainer.n_gpus_per_node': 2, 'trainer.nnodes': 1,
      'trainer.save_freq': 1 if smoke else 25, 'trainer.test_freq': 1 if smoke else 25,
      'trainer.project_name': 'EvidenceGap', 'trainer.experiment_name': out.name,
      'trainer.total_epochs': 30, 'trainer.total_training_steps': 2 if smoke else max(c['steps'])+1,
      'trainer.default_hdfs_dir': None, 'trainer.default_local_dir': str(out),
      'trainer.er_entropy_calibration.enable': False,
      'max_turns': c['max_searches'], 'retriever.url': c['retriever'], 'retriever.topk': 3,
    }
    def val(v):
        if v is None: return 'null'
        if isinstance(v, bool): return str(v).lower()
        return str(v)
    return [c['training_python'], '-m', 'verl.trainer.main_ppo'] + [f'{k}={val(v)}' for k,v in opts.items()]


def prepare(c):
    import pandas as pd
    out = Path(c['output']); out.mkdir(parents=True, exist_ok=True)
    frozen(out/'protocol.resolved.json', c)
    train = pd.read_parquet(c['train_file']); test = pd.read_parquet(c['test_file'])
    old = pd.read_parquet(c['old_validation_file'])
    train = train[train.data_source == 'hotpotqa']
    # Exclude normalized question overlap, including any previously used diagnostic validation.
    excluded = set(map(normalize, train.question)) | set(map(normalize, old.question))
    pool = test[(test.data_source == 'hotpotqa') & ~test.question.map(normalize).isin(excluded)].copy()
    pool['_q'] = pool.question.map(normalize); pool = pool.drop_duplicates('_q').drop(columns='_q')
    pool = pool.sort_values('question').sample(frac=1, random_state=c['split_seed'])
    sizes = [c['diagnostic_questions'], c['evaluation_questions'], c['development_questions']]
    if len(pool) < sum(sizes): raise ValueError('Insufficient disjoint HotpotQA questions')
    panels = {}; start = 0
    for name,n in zip(('diagnostic','evaluation','development'),sizes):
        frame = pool.iloc[start:start+n]; start += n
        panels[name] = [dict(qid=str(r['id']), question=r['question'], gold=list(r['golden_answers']),
                             prompt=list(r['prompt'])) for r in frame.to_dict('records')]
        if name == 'development' and not (out/'development.parquet').exists():
            frame.to_parquet(out/'development.parquet',index=False)
    frozen(out/'panels.json', panels)
    save(out/'split_audit.json', dict(train_hotpotqa=len(train), eligible=len(pool), sizes=sizes,
         disjoint_normalized_questions=True, old_validation_excluded=True,
         scope='Student-training heldout; the frozen teacher may have prior benchmark exposure'))
    print(f'Prepared disjoint HotpotQA panels in {out}', flush=True)


def training_source(c):
    """Freeze experiment source separately; never patch code used by live jobs."""
    import shutil
    import subprocess
    dest=Path(c['output'])/'training_source'
    if (dest/'SNAPSHOT.json').exists(): return dest
    if dest.exists(): raise RuntimeError(f'Incomplete source snapshot: {dest}; inspect before restarting')
    dest.mkdir(parents=True)
    for name in ('verl','search_r1'):
        shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    path=dest/'search_r1/llm_agent/generation.py'
    original=path.read_text()
    old='token_start < span_end and token_end > span_start'
    new='token_start >= span_start and token_end <= span_end'
    if original.count(old)!=1: raise RuntimeError('Training tokenizer source changed; review boundary patch before using this snapshot')
    path.write_text(original.replace(old,new))
    # Keep the exact source change reviewable beside the otherwise unmodified snapshot.
    import difflib
    patch=''.join(difflib.unified_diff(original.splitlines(True),path.read_text().splitlines(True),
                   fromfile='original/search_r1/llm_agent/generation.py',tofile='snapshot/search_r1/llm_agent/generation.py'))
    (dest/'OBSERVATION_BOUNDARY.patch').write_text(patch)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save(dest/'SNAPSHOT.json',dict(base_commit=commit,source=str(ROOT),
        includes_current_working_tree=True,patch='Preserve tokens straddling evidence/protocol boundaries',
        live_source_modified=False))
    return dest
