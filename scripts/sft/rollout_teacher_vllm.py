import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import aiohttp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from search_r1.sft.trajectory_utils import (
    INVALID_ACTION_FEEDBACK,
    build_prompt_messages,
    ensure_action_closed,
    evaluate_trajectory_quality,
    extract_search_queries,
    format_information_observation,
    format_retrieval_documents,
    normalize_question,
    truncate_to_first_action,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Roll out Search-R1 teacher trajectories through a deployed vLLM server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path used to apply the chat template.")
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument(
        "--input-parquet",
        default=None,
        help="Optional local training parquet. When set, samples are selected from this file instead of FlashRAG.",
    )
    parser.add_argument("--data-sources", default="nq,hotpotqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples-per-source", default="5000",
                        help="Samples per data source. A single int (same for all) or comma-separated ints (one per source, matching --data-sources order).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="data/teacher_rollout")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-invalid-actions", type=int, default=2)
    parser.add_argument("--min-search-turns", type=int, default=1)
    parser.add_argument("--max-search-turns", type=int, default=4)
    parser.add_argument("--require-retrieval-hit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_split(dataset_dict, requested_split):
    if requested_split in dataset_dict:
        return requested_split
    for candidate in ("train", "test", "dev", "validation"):
        if candidate in dataset_dict:
            return candidate
    raise ValueError(f"Unable to resolve split for dataset keys: {list(dataset_dict.keys())}")


def parse_per_source_counts(samples_per_source: str, data_sources: Sequence[str]) -> List[int]:
    """Parse --samples-per-source into a list of ints, one per data source.

    Accepts a single int (applied to all sources) or a comma-separated list
    (one per source, same order as --data-sources).
    """
    parts = [s.strip() for s in samples_per_source.split(",")]
    if len(parts) == 1:
        return [int(parts[0])] * len(data_sources)
    counts = [int(p) for p in parts]
    if len(counts) != len(data_sources):
        raise ValueError(
            f"--samples-per-source has {len(counts)} values but "
            f"--data-sources has {len(data_sources)} ({data_sources}). "
            "Either pass a single int (same for all) or one int per source."
        )
    return counts


def load_samples(data_sources: Sequence[str], split: str, samples_per_source: str, seed: int,
                 input_parquet: str | None = None) -> List[Dict]:
    per_source_counts = parse_per_source_counts(samples_per_source, data_sources)
    samples = []

    if input_parquet:
        dataset = load_dataset("parquet", data_files={split: input_parquet}, split=split)
        for data_source, count in zip(data_sources, per_source_counts):
            source_idx = 0
            for row_idx, example in enumerate(dataset):
                if example.get("data_source") != data_source:
                    continue
                question = normalize_question(example["question"])
                prompt = example.get("prompt") or build_prompt_messages(question)
                ground_truth = example.get("reward_model", {}).get("ground_truth")
                if not ground_truth:
                    ground_truth = {"target": example["golden_answers"]}
                example_split = example.get("extra_info", {}).get("split", split)
                example_id = example.get("id", row_idx)
                samples.append({
                    "sample_id": f"{data_source}-{example_split}-{example_id}",
                    "data_source": data_source,
                    "split": example_split,
                    "question": question,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                })
                source_idx += 1
                if source_idx >= count:
                    break
        return samples

    for offset, (data_source, count) in enumerate(zip(data_sources, per_source_counts)):
        dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", data_source)
        resolved_split = resolve_split(dataset, split)
        split_dataset = dataset[resolved_split].shuffle(seed=seed + offset)
        limit = min(count, len(split_dataset))
        for idx in range(limit):
            example = split_dataset[idx]
            question = normalize_question(example["question"])
            prompt = build_prompt_messages(question)
            samples.append({
                "sample_id": f"{data_source}-{resolved_split}-{idx}",
                "data_source": data_source,
                "split": resolved_split,
                "question": question,
                "prompt": prompt,
                "ground_truth": {"target": example["golden_answers"]},
            })
    return samples


def load_processed_ids(raw_output_path: str) -> set:
    if not os.path.exists(raw_output_path):
        return set()
    processed = set()
    with open(raw_output_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            processed.add(record["sample_id"])
    return processed


class VLLMClient:
    def __init__(self, base_url: str, model: str, max_new_tokens: int, temperature: float, top_p: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    async def complete(self, session: aiohttp.ClientSession, prompt_text: str) -> Dict:
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": 1,
            "stream": False,
            "include_stop_str_in_output": True,
            "stop": ["</search>", "</answer>"],
        }
        async with session.post(f"{self.base_url}/completions", json=payload) as response:
            response.raise_for_status()
            return await response.json()


class RetrieverClient:
    def __init__(self, retriever_url: str, topk: int):
        self.retriever_url = retriever_url
        self.topk = topk

    async def search(self, session: aiohttp.ClientSession, query: str) -> List[Dict]:
        payload = {"queries": [query], "topk": self.topk, "return_scores": True}
        async with session.post(self.retriever_url, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
        return data["result"][0]


async def rollout_one_sample(sample: Dict,
                             tokenizer,
                             llm_client: VLLMClient,
                             retriever_client: RetrieverClient,
                             session: aiohttp.ClientSession,
                             args) -> Dict:
    start_time = time.time()
    prompt_text = sample["prompt"][0]["content"]
    if tokenizer.chat_template:
        rendered_prompt = tokenizer.apply_chat_template(sample["prompt"], add_generation_prompt=True, tokenize=False)
    else:
        rendered_prompt = prompt_text

    context_text = rendered_prompt
    trajectory_parts: List[str] = []
    invalid_action_count = 0
    stop_reason = "max_turns"
    usage_totals = defaultdict(int)
    search_logs = []
    request_errors = []

    for step_idx in range(args.max_turns):
        try:
            completion = await llm_client.complete(session, context_text)
        except Exception as exc:  # noqa: BLE001
            stop_reason = "llm_error"
            request_errors.append(f"llm:{exc}")
            break

        choice = completion["choices"][0]
        output_text = ensure_action_closed(choice.get("text", ""))
        output_text, action = truncate_to_first_action(output_text)

        usage = completion.get("usage", {})
        for key, value in usage.items():
            if isinstance(value, int):
                usage_totals[key] += value

        if action is None:
            trajectory_parts.append(output_text)
            trajectory_parts.append(INVALID_ACTION_FEEDBACK)
            context_text += output_text + INVALID_ACTION_FEEDBACK
            invalid_action_count += 1
            if invalid_action_count > args.max_invalid_actions:
                stop_reason = "too_many_invalid_actions"
                break
            continue

        trajectory_parts.append(output_text)
        if action == "search":
            query = extract_search_queries(output_text)[-1]
            try:
                retrieval_result = await retriever_client.search(session, query)
            except Exception as exc:  # noqa: BLE001
                stop_reason = "retriever_error"
                request_errors.append(f"retriever:{exc}")
                break

            search_results = format_retrieval_documents(retrieval_result)
            observation = format_information_observation(search_results)
            trajectory_parts.append(observation)
            context_text += output_text + observation
            search_logs.append({
                "turn_index": step_idx,
                "query": query,
                "formatted_results": search_results,
                "num_results": len(retrieval_result),
            })
            continue

        if action == "answer":
            context_text += output_text
            stop_reason = "answer"
            break

    trajectory_text = "".join(trajectory_parts)
    quality = evaluate_trajectory_quality(
        trajectory_text=trajectory_text,
        ground_truth=sample["ground_truth"],
        invalid_action_count=invalid_action_count,
        min_search_turns=args.min_search_turns,
        max_search_turns=args.max_search_turns,
        require_retrieval_hit=args.require_retrieval_hit,
    )

    return {
        "sample_id": sample["sample_id"],
        "data_source": sample["data_source"],
        "split": sample["split"],
        "question": sample["question"],
        "prompt": sample["prompt"],
        "prompt_text": prompt_text,
        "rendered_prompt": rendered_prompt,
        "ground_truth": sample["ground_truth"],
        "trajectory_text": trajectory_text,
        "response": trajectory_text,
        "search_logs": search_logs,
        "stop_reason": stop_reason,
        "request_errors": request_errors,
        "elapsed_seconds": round(time.time() - start_time, 4),
        "usage": dict(usage_totals),
        **quality,
    }


async def worker(name: str,
                 queue: asyncio.Queue,
                 tokenizer,
                 llm_client: VLLMClient,
                 retriever_client: RetrieverClient,
                 session: aiohttp.ClientSession,
                 writer_lock: asyncio.Lock,
                 raw_output_path: str,
                 accepted_output_path: str,
                 progress: tqdm,
                 stats: Counter,
                 args) -> None:
    while True:
        sample = await queue.get()
        if sample is None:
            queue.task_done()
            return

        record = await rollout_one_sample(sample, tokenizer, llm_client, retriever_client, session, args)
        line = json.dumps(record, ensure_ascii=False)

        async with writer_lock:
            with open(raw_output_path, "a", encoding="utf-8") as raw_fout:
                raw_fout.write(line + "\n")
            if record["quality_pass"]:
                with open(accepted_output_path, "a", encoding="utf-8") as accepted_fout:
                    accepted_fout.write(line + "\n")

        stats["processed"] += 1
        stats[f"source::{record['data_source']}"] += 1
        stats[f"stop::{record['stop_reason']}"] += 1
        if record["quality_pass"]:
            stats["quality_pass"] += 1
        if record["answer_correct"]:
            stats["answer_correct"] += 1
        progress.update(1)
        queue.task_done()


async def main_async(args):
    os.makedirs(args.output_dir, exist_ok=True)
    raw_output_path = os.path.join(args.output_dir, "raw_rollouts.jsonl")
    accepted_output_path = os.path.join(args.output_dir, "accepted_rollouts.jsonl")
    summary_output_path = os.path.join(args.output_dir, "summary.json")

    if args.overwrite:
        for path in (raw_output_path, accepted_output_path, summary_output_path):
            if os.path.exists(path):
                os.remove(path)

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data_sources = [item.strip() for item in args.data_sources.split(",") if item.strip()]
    samples = load_samples(
        data_sources,
        args.split,
        args.samples_per_source,
        args.seed,
        input_parquet=args.input_parquet,
    )
    processed_ids = load_processed_ids(raw_output_path)
    samples = [sample for sample in samples if sample["sample_id"] not in processed_ids]

    llm_client = VLLMClient(
        base_url=args.base_url,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    retriever_client = RetrieverClient(args.retriever_url, args.topk)

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    queue = asyncio.Queue()
    for sample in samples:
        queue.put_nowait(sample)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    writer_lock = asyncio.Lock()
    stats = Counter()
    progress = tqdm(total=len(samples), desc="teacher rollout", dynamic_ncols=True)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                worker(
                    name=f"worker-{idx}",
                    queue=queue,
                    tokenizer=tokenizer,
                    llm_client=llm_client,
                    retriever_client=retriever_client,
                    session=session,
                    writer_lock=writer_lock,
                    raw_output_path=raw_output_path,
                    accepted_output_path=accepted_output_path,
                    progress=progress,
                    stats=stats,
                    args=args,
                )
            )
            for idx in range(args.concurrency)
        ]
        await queue.join()
        await asyncio.gather(*tasks)

    progress.close()
    summary = {
        "model": args.model,
        "tokenizer": tokenizer_name,
        "input_parquet": args.input_parquet,
        "data_sources": data_sources,
        "samples_requested_per_source": dict(
            zip(data_sources, parse_per_source_counts(args.samples_per_source, data_sources))
        ),
        "processed": stats["processed"],
        "quality_pass": stats["quality_pass"],
        "answer_correct": stats["answer_correct"],
        "by_key": dict(stats),
    }
    with open(summary_output_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
