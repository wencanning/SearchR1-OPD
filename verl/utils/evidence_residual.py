"""Evidence-intervention target construction for on-policy distillation."""

import math
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import torch


@dataclass(frozen=True)
class EntropyCalibrationResult:
    temperature: float
    target_entropy: float
    achieved_entropy: float
    entropy_gap: float
    evaluations: int
    stop_reason: str


def select_calibration_ids(
    question_ids: Iterable[str],
    seed: int,
    fraction: float = 0.05,
) -> List[str]:
    """Select a deterministic seeded fraction without depending on input order."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError('fraction must be in (0, 1]')
    unique_ids = sorted({str(question_id) for question_id in question_ids})
    if not unique_ids:
        return []
    ranked = sorted(
        unique_ids,
        key=lambda question_id: (
            zlib.crc32(f'{seed}:{question_id}'.encode('utf-8')),
            question_id,
        ),
    )
    count = max(1, math.ceil(len(ranked) * fraction))
    return ranked[:count]


def calibrate_entropy_matched_temperature(
    mean_entropy_at_temperature: Callable[[float], float],
    target_entropy: float,
    entropy_tolerance: float = 1e-6,
    relative_temperature_tolerance: float = 1e-8,
    max_iterations: int = 100,
    max_expansions: int = 20,
) -> EntropyCalibrationResult:
    """Find one positive global temperature with a bounded Brent root solve.

    Optimization is performed in log-temperature coordinates.  The initial
    interval is expanded automatically until the monotone entropy curve
    brackets the requested target.
    """
    if not math.isfinite(target_entropy):
        raise ValueError('target_entropy must be finite')
    if entropy_tolerance <= 0 or relative_temperature_tolerance <= 0:
        raise ValueError('calibration tolerances must be positive')

    cache = {}

    def objective(log_temperature: float) -> float:
        if log_temperature not in cache:
            entropy = float(mean_entropy_at_temperature(math.exp(log_temperature)))
            if not math.isfinite(entropy):
                raise FloatingPointError('calibration entropy callback returned a non-finite value')
            cache[log_temperature] = entropy - target_entropy
        return cache[log_temperature]

    lower = -2.0
    upper = 2.0
    f_lower = objective(lower)
    f_upper = objective(upper)
    width = upper - lower
    for _ in range(max_expansions):
        if f_lower <= 0.0 <= f_upper:
            break
        if f_lower > 0.0:
            upper, f_upper = lower, f_lower
            lower -= width
            f_lower = objective(lower)
        else:
            lower, f_lower = upper, f_upper
            upper += width
            f_upper = objective(upper)
        width *= 2.0
    else:
        raise ValueError('could not bracket target entropy with a positive temperature')

    if abs(f_lower) <= entropy_tolerance:
        root = lower
        stop_reason = 'entropy_tolerance'
    elif abs(f_upper) <= entropy_tolerance:
        root = upper
        stop_reason = 'entropy_tolerance'
    else:
        # Brent-Dekker on the bracketed monotone root in log-temperature.
        a, b = lower, upper
        fa, fb = f_lower, f_upper
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        c, fc = a, fa
        d = c
        used_bisection = True
        stop_reason = 'max_iterations'
        root = b
        for _ in range(max_iterations):
            if fa != fc and fb != fc:
                proposal = (
                    a * fb * fc / ((fa - fb) * (fa - fc))
                    + b * fa * fc / ((fb - fa) * (fb - fc))
                    + c * fa * fb / ((fc - fa) * (fc - fb))
                )
            else:
                proposal = b - fb * (b - a) / (fb - fa)

            midpoint_bound = (3.0 * a + b) / 4.0
            outside = not (min(midpoint_bound, b) < proposal < max(midpoint_bound, b))
            slow_after_bisection = used_bisection and abs(proposal - b) >= abs(b - c) / 2.0
            slow_after_interpolation = (not used_bisection) and abs(proposal - b) >= abs(c - d) / 2.0
            tiny_bisection_step = used_bisection and abs(b - c) < relative_temperature_tolerance
            tiny_interpolation_step = (not used_bisection) and abs(c - d) < relative_temperature_tolerance
            if outside or slow_after_bisection or slow_after_interpolation or tiny_bisection_step or tiny_interpolation_step:
                proposal = (a + b) / 2.0
                used_bisection = True
            else:
                used_bisection = False

            f_proposal = objective(proposal)
            d, c = c, b
            if fa * f_proposal < 0.0:
                b, fb = proposal, f_proposal
            else:
                a, fa = proposal, f_proposal
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            root = b

            if abs(fb) <= entropy_tolerance:
                stop_reason = 'entropy_tolerance'
                break
            tau_a = math.exp(a)
            tau_b = math.exp(b)
            if abs(tau_a - tau_b) / min(tau_a, tau_b) <= relative_temperature_tolerance:
                stop_reason = 'relative_temperature_tolerance'
                break

    temperature = math.exp(root)
    achieved_entropy = float(mean_entropy_at_temperature(temperature))
    return EntropyCalibrationResult(
        temperature=temperature,
        target_entropy=target_entropy,
        achieved_entropy=achieved_entropy,
        entropy_gap=abs(achieved_entropy - target_entropy),
        evaluations=len(cache),
        stop_reason=stop_reason,
    )


def build_evidence_hidden_attention_mask(
    attention_mask: torch.Tensor,
    evidence_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an additive causal mask that hides retrieved-content key columns.

    Input slots and position ids are unchanged.  Only valid keys marked by
    ``evidence_mask`` receive the minimum representable additive bias.  The
    shape is accepted directly by Hugging Face Qwen2's SDPA implementation.
    """
    if attention_mask.ndim != 2 or evidence_mask.shape != attention_mask.shape:
        raise ValueError('attention_mask and evidence_mask must have matching [batch, sequence] shapes')
    if not dtype.is_floating_point:
        raise ValueError(f'additive attention masks require a floating dtype, got {dtype}')

    valid_keys = attention_mask.bool()
    hidden_keys = evidence_mask.bool()
    if torch.any(hidden_keys & ~valid_keys):
        raise ValueError('evidence_mask cannot mark padding positions')

    batch_size, sequence_length = attention_mask.shape
    device = attention_mask.device
    blocked_value = torch.finfo(dtype).min
    query_positions = torch.arange(sequence_length, device=device).view(sequence_length, 1)
    key_positions = torch.arange(sequence_length, device=device).view(1, sequence_length)
    future_keys = key_positions > query_positions
    blocked_keys = (~valid_keys | hidden_keys)[:, None, None, :]
    blocked = future_keys[None, None, :, :] | blocked_keys

    additive_mask = torch.zeros(
        (batch_size, 1, sequence_length, sequence_length),
        dtype=dtype,
        device=device,
    )
    return additive_mask.masked_fill(blocked, blocked_value)


def _distribution_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if not torch.isfinite(logits).all():
        raise FloatingPointError('teacher target logits contain non-finite values')
    # Use one normalization for every statistic.  Recomputing the distribution
    # independently with logsumexp and softmax can differ by more than a fixed
    # 1e-6 FP32 reduction tolerance for a large vocabulary, even when both
    # results are valid.  It also turns the per-chunk check into a CUDA sync.
    log_probabilities = torch.log_softmax(logits, dim=-1)
    selected_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    log_prob = log_probabilities.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    if not torch.isfinite(log_prob).all() or not torch.isfinite(entropy).all():
        raise FloatingPointError('teacher target statistics contain non-finite values')
    return {
        'log_prob': log_prob,
        'entropy': entropy,
        'selected_logit': selected_logits,
    }


def compute_evidence_residual_target_stats(
    observed_logits: torch.Tensor,
    hidden_logits: torch.Tensor,
    labels: torch.Tensor,
    token_chunk_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """Compute ``softmax(2 * z_observed - z_hidden)`` in FP32.

    The vocabulary dimension is always complete.  Chunking is only over token
    rows, limiting peak memory without changing the normalization.
    """
    if observed_logits.shape != hidden_logits.shape:
        raise ValueError('observed and hidden logits must have identical shapes')
    if observed_logits.shape[:-1] != labels.shape:
        raise ValueError('labels must match the batch and token dimensions of teacher logits')
    if token_chunk_size <= 0:
        raise ValueError('token_chunk_size must be positive')

    vocab_size = observed_logits.shape[-1]
    flat_observed = observed_logits.reshape(-1, vocab_size)
    flat_hidden = hidden_logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    collected = {
        'target_log_prob': [],
        'target_entropy': [],
        'observed_log_prob': [],
        'observed_entropy': [],
        'hidden_log_prob': [],
        'hidden_entropy': [],
        'selected_logit_delta': [],
    }

    for start in range(0, flat_labels.numel(), token_chunk_size):
        end = min(start + token_chunk_size, flat_labels.numel())
        observed = flat_observed[start:end].float()
        hidden = flat_hidden[start:end].float()
        chunk_labels = flat_labels[start:end]
        target = 2.0 * observed - hidden

        target_stats = _distribution_stats(target, chunk_labels)
        observed_stats = _distribution_stats(observed, chunk_labels)
        hidden_stats = _distribution_stats(hidden, chunk_labels)
        collected['target_log_prob'].append(target_stats['log_prob'])
        collected['target_entropy'].append(target_stats['entropy'])
        collected['observed_log_prob'].append(observed_stats['log_prob'])
        collected['observed_entropy'].append(observed_stats['entropy'])
        collected['hidden_log_prob'].append(hidden_stats['log_prob'])
        collected['hidden_entropy'].append(hidden_stats['entropy'])
        collected['selected_logit_delta'].append(
            observed_stats['selected_logit'] - hidden_stats['selected_logit']
        )

    output_shape = labels.shape
    return {
        key: torch.cat(parts, dim=0).reshape(output_shape)
        for key, parts in collected.items()
    }


def compute_temperature_target_stats(
    observed_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    token_chunk_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """Compute a fixed-temperature visible-teacher control in FP32."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if observed_logits.shape[:-1] != labels.shape:
        raise ValueError('labels must match the batch and token dimensions of teacher logits')
    if token_chunk_size <= 0:
        raise ValueError('token_chunk_size must be positive')

    vocab_size = observed_logits.shape[-1]
    flat_observed = observed_logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    log_probs = []
    entropies = []
    for start in range(0, flat_labels.numel(), token_chunk_size):
        end = min(start + token_chunk_size, flat_labels.numel())
        stats = _distribution_stats(
            flat_observed[start:end].float() / temperature,
            flat_labels[start:end],
        )
        log_probs.append(stats['log_prob'])
        entropies.append(stats['entropy'])

    return {
        'target_log_prob': torch.cat(log_probs, dim=0).reshape(labels.shape),
        'target_entropy': torch.cat(entropies, dim=0).reshape(labels.shape),
    }
