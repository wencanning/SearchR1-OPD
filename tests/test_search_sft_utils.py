import unittest

from search_r1.sft.trajectory_utils import (
    build_parquet_rows,
    build_search_prompt,
    evaluate_trajectory_quality,
    normalize_question,
    split_train_val_rows,
)


class SearchSFTUtilsTests(unittest.TestCase):
    def test_normalize_question_appends_question_mark(self):
        self.assertEqual(normalize_question("Who won"), "Who won?")

    def test_build_search_prompt_contains_agent_protocol(self):
        prompt = build_search_prompt("Who won?")
        self.assertIn("<think>", prompt)
        self.assertIn("<search>", prompt)
        self.assertIn("<information>", prompt)
        self.assertIn("<answer>", prompt)

    def test_evaluate_trajectory_quality_accepts_valid_correct_trajectory(self):
        trajectory = (
            "<think>Need evidence.</think>"
            "<search>championship winner</search>"
            "<information>Doc 1(Title: Finals) Team A won the championship.</information>"
            "<think>I have enough information.</think>"
            "<answer>Team A</answer>"
        )
        metrics = evaluate_trajectory_quality(
            trajectory_text=trajectory,
            ground_truth={"target": ["Team A"]},
            invalid_action_count=0,
            min_search_turns=1,
            max_search_turns=3,
            require_retrieval_hit=True,
        )

        self.assertTrue(metrics["format_valid"])
        self.assertTrue(metrics["answer_correct"])
        self.assertTrue(metrics["retrieval_hit"])
        self.assertTrue(metrics["quality_pass"])
        self.assertEqual(metrics["search_turn_count"], 1)

    def test_build_parquet_rows_keeps_best_record_per_question(self):
        prompt = [{"role": "user", "content": build_search_prompt("Who won?")}]
        records = [
            {
                "sample_id": "nq-train-1",
                "question": "Who won?",
                "data_source": "nq",
                "prompt": prompt,
                "trajectory_text": "<think>x</think><answer>Wrong</answer>",
                "ground_truth": {"target": ["Team A"]},
                "quality_score": 10,
                "quality_pass": False,
            },
            {
                "sample_id": "nq-train-1-retry",
                "question": "Who won?",
                "data_source": "nq",
                "prompt": prompt,
                "trajectory_text": "<think>y</think><answer>Team A</answer>",
                "ground_truth": {"target": ["Team A"]},
                "quality_score": 120,
                "quality_pass": True,
            },
        ]

        rows = build_parquet_rows(records, require_quality_pass=True, dedupe_key="question")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["response"], "<think>y</think><answer>Team A</answer>")
        self.assertEqual(rows[0]["extra_info"]["question"], "Who won?")

    def test_split_train_val_rows_preserves_sources(self):
        rows = [
            {"data_source": "nq", "response": "a"},
            {"data_source": "nq", "response": "b"},
            {"data_source": "hotpotqa", "response": "c"},
            {"data_source": "hotpotqa", "response": "d"},
        ]
        train_rows, val_rows = split_train_val_rows(rows, val_ratio=0.5, seed=7)

        self.assertEqual(len(train_rows), 2)
        self.assertEqual(len(val_rows), 2)
        self.assertEqual(sorted(row["data_source"] for row in val_rows), ["hotpotqa", "nq"])


if __name__ == "__main__":
    unittest.main()
