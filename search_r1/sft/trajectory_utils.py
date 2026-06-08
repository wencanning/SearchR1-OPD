import json
import random
import re
import string
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SEARCH_PROMPT_TEMPLATE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

INVALID_ACTION_FEEDBACK = (
    "\nMy previous action is invalid. "
    "If I want to search, I should put the query between <search> and </search>. "
    "If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
)

ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
INFO_PATTERN = re.compile(r"<information>(.*?)</information>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)
TAG_SPLIT_PATTERN = re.compile(r"(</?(?:think|search|information|answer)>)")


def normalize_question(question: str) -> str:
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"
    return question


def build_search_prompt(question: str) -> str:
    return SEARCH_PROMPT_TEMPLATE.format(question=normalize_question(question))


def build_prompt_messages(question: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": build_search_prompt(question)}]


def lower_no_punc(text: str) -> str:
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text.lower() if ch not in exclude)
    return " ".join(text.split())


def em_check(prediction: str, golden_answers: Sequence[str]) -> bool:
    normalized_prediction = lower_no_punc(prediction)
    for golden_answer in golden_answers:
        if lower_no_punc(golden_answer) == normalized_prediction:
            return True
    return False


def contains_answer(text: str, golden_answers: Sequence[str]) -> bool:
    normalized_text = lower_no_punc(text)
    for golden_answer in golden_answers:
        if lower_no_punc(golden_answer) in normalized_text:
            return True
    return False


def extract_final_answer(trajectory_text: str) -> Optional[str]:
    matches = list(ANSWER_PATTERN.finditer(trajectory_text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_search_queries(trajectory_text: str) -> List[str]:
    return [match.group(1).strip() for match in SEARCH_PATTERN.finditer(trajectory_text)]


def extract_information_blocks(trajectory_text: str) -> List[str]:
    return [match.group(1).strip() for match in INFO_PATTERN.finditer(trajectory_text)]


def count_invalid_action_feedback(trajectory_text: str) -> int:
    return trajectory_text.count("My previous action is invalid.")


def ensure_action_closed(text: str) -> str:
    for tag in ("search", "answer"):
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        last_open = text.rfind(open_tag)
        last_close = text.rfind(close_tag)
        if last_open != -1 and last_open > last_close:
            return text + close_tag
    return text


def truncate_to_first_action(text: str) -> Tuple[str, Optional[str]]:
    match = ACTION_PATTERN.search(text)
    if match is None:
        return text, None
    return text[:match.end()], match.group(1)


def format_retrieval_documents(retrieval_result: Sequence[Dict]) -> str:
    formatted = []
    for idx, doc_item in enumerate(retrieval_result, start=1):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        formatted.append(f"Doc {idx}(Title: {title}) {text}")
    return "\n".join(formatted)


def format_information_observation(search_results: str) -> str:
    return f"\n\n<information>{search_results}</information>\n\n"


def is_valid_trajectory_format(trajectory_text: str) -> Tuple[bool, str]:
    parts = [part for part in TAG_SPLIT_PATTERN.split(trajectory_text) if part and part.strip()]
    state = "start"

    for part in parts:
        if re.fullmatch(r"</?(?:think|search|information|answer)>", part):
            if part == "<think>" and state in {"start", "after_information"}:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "after_information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state not in {"in_think", "in_search", "in_information", "in_answer"}:
                return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete trajectory, ended in state {state}"
    return True, "ok"


def evaluate_trajectory_quality(trajectory_text: str,
                                ground_truth: Dict,
                                invalid_action_count: Optional[int] = None,
                                min_search_turns: int = 1,
                                max_search_turns: Optional[int] = None,
                                require_retrieval_hit: bool = False) -> Dict:
    if invalid_action_count is None:
        invalid_action_count = count_invalid_action_feedback(trajectory_text)

    format_valid, format_error = is_valid_trajectory_format(trajectory_text)
    search_queries = extract_search_queries(trajectory_text)
    information_blocks = extract_information_blocks(trajectory_text)
    final_answer = extract_final_answer(trajectory_text)
    retrieval_hit = any(contains_answer(block, ground_truth["target"]) for block in information_blocks)
    answer_correct = final_answer is not None and em_check(final_answer, ground_truth["target"])

    search_turn_count = len(search_queries)
    information_turn_count = len(information_blocks)
    turn_within_limit = max_search_turns is None or search_turn_count <= max_search_turns

    quality_pass = (
        format_valid and
        answer_correct and
        invalid_action_count == 0 and
        search_turn_count >= min_search_turns and
        information_turn_count == search_turn_count and
        turn_within_limit and
        (retrieval_hit if require_retrieval_hit else True)
    )

    quality_score = (
        (100 if answer_correct else 0) +
        (20 if format_valid else 0) +
        (10 if retrieval_hit else 0) +
        min(search_turn_count, 5) -
        (5 * invalid_action_count)
    )

    return {
        "format_valid": format_valid,
        "format_error": format_error,
        "final_answer": final_answer,
        "answer_correct": answer_correct,
        "retrieval_hit": retrieval_hit,
        "search_turn_count": search_turn_count,
        "information_turn_count": information_turn_count,
        "invalid_action_count": invalid_action_count,
        "quality_score": quality_score,
        "quality_pass": quality_pass,
        "search_queries": search_queries,
    }


def build_parquet_rows(records: Iterable[Dict],
                       require_quality_pass: bool = True,
                       dedupe_key: str = "question") -> List[Dict]:
    best_records = {}
    for record in records:
        if require_quality_pass and not record.get("quality_pass", False):
            continue
        key = record.get(dedupe_key) if dedupe_key else record["sample_id"]
        current = best_records.get(key)
        if current is None or record.get("quality_score", 0) > current.get("quality_score", 0):
            best_records[key] = record

    rows = []
    for record in best_records.values():
        rows.append({
            "prompt": record["prompt"],
            "response": record["trajectory_text"],
            "data_source": record["data_source"],
            "ground_truth": record["ground_truth"],
            "extra_info": {
                "sample_id": record["sample_id"],
                "question": record["question"],
                "quality_score": record.get("quality_score", 0),
                "answer_correct": record.get("answer_correct", False),
                "retrieval_hit": record.get("retrieval_hit", False),
                "search_turn_count": record.get("search_turn_count", 0),
            },
        })
    return rows


def split_train_val_rows(rows: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not rows:
        return [], []

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["data_source"]].append(row)

    train_rows = []
    val_rows = []
    rng = random.Random(seed)
    for source_rows in grouped.values():
        source_rows = list(source_rows)
        rng.shuffle(source_rows)
        val_count = int(round(len(source_rows) * val_ratio))
        if val_ratio > 0 and val_count == 0 and len(source_rows) > 1:
            val_count = 1
        if val_count >= len(source_rows) and len(source_rows) > 1:
            val_count = len(source_rows) - 1
        val_rows.extend(source_rows[:val_count])
        train_rows.extend(source_rows[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def dump_jsonl(records: Iterable[Dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
