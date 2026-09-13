import gzip
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


TAG_PATTERN = re.compile(r"</?(?:think|search|information|answer)>")
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
INFO_PATTERN = re.compile(r"<information>(.*?)</information>", re.DOTALL)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except Exception:
        return getattr(config, key, default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _round_float(value: float, digits: int) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _float_list(tensor: Optional[torch.Tensor], length: int, digits: int) -> Optional[List[float]]:
    if tensor is None:
        return None
    values = tensor[:length].detach().cpu().float().tolist()
    return [_round_float(v, digits) for v in values]


def _int_list(tensor: torch.Tensor, length: int) -> List[int]:
    return [int(v) for v in tensor[:length].detach().cpu().tolist()]


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _extract_answer(text: str) -> Optional[str]:
    matches = list(ANSWER_PATTERN.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _ground_truth_targets(reward_model: Any) -> List[str]:
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            target = ground_truth.get("target", [])
        else:
            target = ground_truth
        if isinstance(target, str):
            return [target]
        if isinstance(target, (list, tuple, np.ndarray)):
            return [str(x) for x in target]
    return []


def _answer_correct(answer: Optional[str], targets: List[str]) -> Optional[bool]:
    if answer is None or not targets:
        return None
    normalized = _normalize_answer(answer)
    return any(normalized == _normalize_answer(target) for target in targets)


def _retrieval_hit(text: str, targets: List[str]) -> Optional[bool]:
    if not targets:
        return None
    blocks = INFO_PATTERN.findall(text)
    if not blocks:
        return False
    normalized_blocks = [_normalize_answer(block) for block in blocks]
    return any(
        _normalize_answer(target) in block
        for target in targets
        for block in normalized_blocks
    )


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def infer_token_segments(token_texts: List[str]) -> tuple[List[str], List[int]]:
    """Infer coarse Search-R1 segment labels from decoded token text."""
    text = "".join(token_texts)
    spans = []
    cursor = 0
    for token_text in token_texts:
        start = cursor
        cursor += len(token_text)
        spans.append((start, cursor))

    tags = list(TAG_PATTERN.finditer(text))
    segments = []
    turn_ids = []
    tag_idx = 0
    state = "outside"
    turn = 0

    for start, end in spans:
        midpoint = (start + end) / 2
        while tag_idx < len(tags) and tags[tag_idx].end() <= midpoint:
            tag = tags[tag_idx].group(0)
            if tag == "<think>":
                state = "think"
            elif tag == "</think>":
                state = "after_think"
            elif tag == "<search>":
                turn += 1
                state = "search"
            elif tag == "</search>":
                state = "after_search"
            elif tag == "<information>":
                state = "information"
            elif tag == "</information>":
                state = "post_information"
            elif tag == "<answer>":
                state = "answer"
            elif tag == "</answer>":
                state = "after_answer"
            tag_idx += 1

        overlaps_tag = any(tag.start() < end and start < tag.end() for tag in tags)
        segments.append("tag" if overlaps_tag else state)
        turn_ids.append(turn)

    return segments, turn_ids


def infer_evidence_step_ids(token_texts: List[str]) -> List[int]:
    """Infer step ids by counting completed information blocks before each token."""
    text = "".join(token_texts)
    spans = []
    cursor = 0
    for token_text in token_texts:
        start = cursor
        cursor += len(token_text)
        spans.append((start, cursor))

    tags = list(TAG_PATTERN.finditer(text))
    step_ids = []
    tag_idx = 0
    step_id = 0
    for start, end in spans:
        midpoint = (start + end) / 2
        while tag_idx < len(tags) and tags[tag_idx].end() <= midpoint:
            if tags[tag_idx].group(0) == "</information>":
                step_id += 1
            tag_idx += 1
        step_ids.append(step_id)
    return step_ids


def retrieval_hits_by_information_block(text: str, targets: List[str]) -> Optional[List[bool]]:
    if not targets:
        return None
    blocks = INFO_PATTERN.findall(text)
    normalized_targets = [_normalize_answer(target) for target in targets]
    hits = []
    for block in blocks:
        normalized_block = _normalize_answer(block)
        hits.append(any(target in normalized_block for target in normalized_targets))
    return hits


class OPDUncertaintyDumper:
    """Persist compressed OPD uncertainty traces for offline analysis."""

    def __init__(self, tokenizer: Any, trainer_config: Any, opd_config: Any):
        diagnostics_config = _cfg_get(opd_config, "diagnostics", {})
        default_local_dir_value = _cfg_get(trainer_config, "default_local_dir", None)
        default_local_dir = (
            "opd_diagnostics"
            if default_local_dir_value is None
            else os.path.expanduser(str(default_local_dir_value))
        )
        output_dir = _cfg_get(diagnostics_config, "output_dir", None)
        if output_dir is None:
            output_dir = (
                default_local_dir
                if Path(default_local_dir).name == "opd_diagnostics"
                else os.path.join(default_local_dir, "opd_diagnostics")
            )

        self.tokenizer = tokenizer
        self.output_dir = Path(os.path.expanduser(str(output_dir)))
        self.every_n_steps = int(_cfg_get(diagnostics_config, "every_n_steps", 1))
        self.max_sequences_per_step = int(_cfg_get(diagnostics_config, "max_sequences_per_step", 64))
        self.float_precision = int(_cfg_get(diagnostics_config, "float_precision", 4))
        self.sample_strategy = str(_cfg_get(diagnostics_config, "sample_strategy", "random"))
        self.include_token_text = bool(_cfg_get(diagnostics_config, "include_token_text", True))
        self.include_decoded_response = bool(_cfg_get(diagnostics_config, "include_decoded_response", True))
        self.top_entropy_tokens = int(_cfg_get(diagnostics_config, "top_entropy_tokens", 32))
        self.compress = bool(_cfg_get(diagnostics_config, "compress", False))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_dump(self, global_step: int) -> bool:
        return self.every_n_steps > 0 and global_step % self.every_n_steps == 0

    def _select_indices(self, batch: Any, opd_mask: torch.Tensor, global_step: int) -> List[int]:
        batch_size = len(batch)
        limit = min(self.max_sequences_per_step, batch_size)
        indices = list(range(batch_size))
        if limit >= batch_size:
            return indices

        if self.sample_strategy == "entropy_top" and "ref_entropy" in batch.batch:
            entropy = batch.batch["ref_entropy"].detach().cpu().float()
            mask = opd_mask.detach().cpu().float()
            means = (entropy * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
            return torch.topk(means, k=limit).indices.tolist()

        if self.sample_strategy == "head":
            return indices[:limit]

        rng = random.Random(global_step)
        return sorted(rng.sample(indices, limit))

    def dump(self, batch: Any, opd_mask: torch.Tensor, global_step: int, epoch: int, metrics: Dict[str, float]) -> Path:
        suffix = "jsonl.gz" if self.compress else "jsonl"
        path = self.output_dir / f"step_{global_step:06d}.{suffix}"
        selected_indices = self._select_indices(batch, opd_mask, global_step)
        response_length = batch.batch["responses"].shape[-1]
        response_mask = batch.batch["attention_mask"][:, -response_length:]

        opener = gzip.open if self.compress else open
        with opener(path, "wt", encoding="utf-8") as f:
            header = {
                "record_type": "metadata",
                "global_step": global_step,
                "epoch": epoch,
                "selected_sequences": len(selected_indices),
                "batch_size": len(batch),
                "sample_strategy": self.sample_strategy,
                "metrics": {k: _json_safe(v) for k, v in metrics.items()},
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")

            for seq_idx in selected_indices:
                record = self._make_record(batch, opd_mask, response_mask, seq_idx, global_step, epoch)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return path

    def _make_record(
        self,
        batch: Any,
        opd_mask: torch.Tensor,
        response_mask: torch.Tensor,
        seq_idx: int,
        global_step: int,
        epoch: int,
    ) -> Dict[str, Any]:
        valid_len = int(response_mask[seq_idx].sum().item())
        response_ids = _int_list(batch.batch["responses"][seq_idx], valid_len)
        token_texts = [_decode_token(self.tokenizer, token_id) for token_id in response_ids]
        segments, turn_ids = infer_token_segments(token_texts)
        decoded_response = "".join(token_texts)

        reward_model = None
        if "reward_model" in batch.non_tensor_batch:
            reward_model = batch.non_tensor_batch["reward_model"][seq_idx]
        targets = _ground_truth_targets(reward_model)
        final_answer = _extract_answer(decoded_response)

        tensors = batch.batch
        old_entropy = tensors["old_entropy"][seq_idx] if "old_entropy" in tensors else None
        token_level_scores = tensors["token_level_scores"][seq_idx] if "token_level_scores" in tensors else None
        token_level_rewards = tensors["token_level_rewards"][seq_idx] if "token_level_rewards" in tensors else None

        record = {
            "record_type": "trajectory",
            "global_step": global_step,
            "epoch": epoch,
            "sequence_index": seq_idx,
            "uid": _json_safe(batch.non_tensor_batch.get("uid", [None] * len(batch))[seq_idx]),
            "index": _json_safe(batch.non_tensor_batch.get("index", [None] * len(batch))[seq_idx]),
            "data_source": _json_safe(batch.non_tensor_batch.get("data_source", ["unknown"] * len(batch))[seq_idx]),
            "reward_model": _json_safe(reward_model),
            "ground_truth_targets": targets,
            "final_answer": final_answer,
            "answer_correct": _answer_correct(final_answer, targets),
            "retrieval_hit": _retrieval_hit(decoded_response, targets),
            "sequence_score": _round_float(float(token_level_scores[:valid_len].sum().item()), self.float_precision)
            if token_level_scores is not None else None,
            "sequence_reward": _round_float(float(token_level_rewards[:valid_len].sum().item()), self.float_precision)
            if token_level_rewards is not None else None,
            "valid_length": valid_len,
            "token_ids": response_ids,
            "loss_mask": _int_list(opd_mask[seq_idx], valid_len),
            "segments": segments,
            "turn_ids": turn_ids,
            "teacher_entropy": _float_list(tensors["ref_entropy"][seq_idx], valid_len, self.float_precision),
            "student_entropy": _float_list(old_entropy, valid_len, self.float_precision),
            "teacher_log_prob": _float_list(tensors["ref_log_prob"][seq_idx], valid_len, self.float_precision),
            "student_log_prob": _float_list(tensors["old_log_probs"][seq_idx], valid_len, self.float_precision),
            "logp_gap_teacher_minus_student": _float_list(
                tensors["ref_log_prob"][seq_idx] - tensors["old_log_probs"][seq_idx],
                valid_len,
                self.float_precision,
            ),
            "opd_advantage": _float_list(tensors["opd_advantages"][seq_idx], valid_len, self.float_precision)
            if "opd_advantages" in tensors else None,
            "opd_rce_weight": _float_list(tensors["opd_rce_weights"][seq_idx], valid_len, self.float_precision)
            if "opd_rce_weights" in tensors else None,
            "opd_effective_distillation_coef": _float_list(
                tensors["opd_effective_distillation_coef"][seq_idx], valid_len, self.float_precision
            ) if "opd_effective_distillation_coef" in tensors else None,
            "weighted_opd_advantage": _float_list(
                tensors["weighted_opd_advantages"][seq_idx], valid_len, self.float_precision
            ) if "weighted_opd_advantages" in tensors else None,
            "grpo_advantage": _float_list(tensors["grpo_advantages"][seq_idx], valid_len, self.float_precision)
            if "grpo_advantages" in tensors else None,
            "combined_advantage": _float_list(tensors["advantages"][seq_idx], valid_len, self.float_precision)
            if "advantages" in tensors else None,
        }

        if self.include_token_text:
            record["token_texts"] = token_texts
        if self.include_decoded_response:
            record["decoded_response"] = decoded_response

        teacher_entropy = record["teacher_entropy"] or []
        high_entropy = sorted(
            [
                {
                    "token_index": i,
                    "token_text": token_texts[i] if self.include_token_text else None,
                    "segment": segments[i],
                    "turn_id": turn_ids[i],
                    "teacher_entropy": teacher_entropy[i],
                    "logp_gap_teacher_minus_student": record["logp_gap_teacher_minus_student"][i],
                }
                for i, mask in enumerate(record["loss_mask"])
                if mask and teacher_entropy[i] is not None
            ],
            key=lambda x: x["teacher_entropy"],
            reverse=True,
        )
        record["top_teacher_entropy_tokens"] = high_entropy[:self.top_entropy_tokens]
        record["segment_summary"] = self._segment_summary(record)
        return record

    def _segment_summary(self, record: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
        summary: Dict[str, Dict[str, Optional[float]]] = {}
        teacher_entropy = record.get("teacher_entropy") or []
        student_entropy = record.get("student_entropy") or []
        logp_gap = record.get("logp_gap_teacher_minus_student") or []
        loss_mask = record.get("loss_mask") or []

        for i, segment in enumerate(record.get("segments") or []):
            if i >= len(loss_mask) or not loss_mask[i]:
                continue
            bucket = summary.setdefault(segment, {
                "count": 0,
                "teacher_entropy_sum": 0.0,
                "student_entropy_sum": 0.0,
                "logp_gap_sum": 0.0,
            })
            bucket["count"] += 1
            bucket["teacher_entropy_sum"] += teacher_entropy[i] or 0.0
            bucket["student_entropy_sum"] += student_entropy[i] if i < len(student_entropy) and student_entropy[i] is not None else 0.0
            bucket["logp_gap_sum"] += logp_gap[i] if i < len(logp_gap) and logp_gap[i] is not None else 0.0

        for segment, bucket in summary.items():
            count = bucket["count"]
            bucket["teacher_entropy_mean"] = _round_float(bucket.pop("teacher_entropy_sum") / count, self.float_precision)
            bucket["student_entropy_mean"] = _round_float(bucket.pop("student_entropy_sum") / count, self.float_precision)
            bucket["logp_gap_mean"] = _round_float(bucket.pop("logp_gap_sum") / count, self.float_precision)

        return summary
