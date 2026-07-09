"""
harness.py — Full EvoAgent training loop.

Orchestrates T iterations of:
  1. Propose a new strategy (or use the seed for iteration 0).
  2. Evaluate on the train subset.
  3. Evaluate on the dev split.
  4. Reflect on the results.
  5. Save state to disk.
"""

from __future__ import annotations

import logging
import time
import collections, random as _random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset
from tqdm import tqdm

from src.executor import EvalResult, evaluate, classify_question_type, TokenBudget
from src.model import QwenInference
from src.self_proposer import propose_self
from src.self_reflector import reflect_self
from src.strategy import Strategy, StrategyHistory, StrategyMetadata, make_seed_strategy, CoTFormat

logger = logging.getLogger(__name__)


def select_parent_strategy(
    history: StrategyHistory,
    afo_mode: str,
    afo_prob_best: float = 0.4,
    afo_prob_original: float = 0.3,
    afo_prob_latest: float = 0.3,
) -> Optional[Strategy]:
    """
    Select the parent strategy to mutate based on the Always-From-Original (AFO) mode.

    TODO: Implement the Always-From-Original (AFO) parent selection policy.
    Modes:
    - 'none': Always mutate from the latest strategy in history.
    - 'best': Always mutate from the best strategy (highest dev accuracy) found so far.
    - 'original': Always mutate from the original seed strategy (iteration 0).
    - 'probabilistic': Select from Best, Original, and Latest based on the provided probabilities.
    """
    if not history.strategies:
        return None

    if afo_mode == "none":
        return history.strategies[-1]
    elif afo_mode == "best":
        best_strategy = max(history.strategies, key=lambda s: s.metadata.dev_accuracy or 0)
        return best_strategy
    elif afo_mode == "original":
        return history.strategies[0]
    elif afo_mode == "probabilistic":
        import random
        choices = random.choices(["best", "original", "latest"], weights=[afo_prob_best, afo_prob_original, afo_prob_latest], k=1)
        choice = choices[0]
        if choice == "best":
            best_strategy = max(history.strategies, key=lambda s: s.metadata.dev_accuracy or 0)
            return best_strategy
        elif choice == "original":
            return history.strategies[0]
        else:
            return history.strategies[-1]
    else:
        print("Unknown AFO mode:", afo_mode)
        return None


def select_curriculum_dataset(train_dataset: Dataset, iteration: int, train_size: int) -> Dataset:
    """
    Selects the train_subset dynamically based on curriculum learning.
    - Iteration 1: Easiest, shortest reading passages.
    - Iteration 2: Harder "table_op" or multi-step questions.
    - Other iterations (like 0, 3+): Easiest or standard subset.
    """
    rows = list(train_dataset)
    row_details = []
    for r in rows:
        passage = r.get("context") or r.get("article") or r.get("passage") or ""
        gold_program = r.get("answer") or ""
        
        q_type = classify_question_type(gold_program)
        is_hard = q_type in ["table_op", "division"] or ("," in gold_program)
        
        row_details.append({
            "row": r,
            "passage_len": len(passage),
            "is_hard": is_hard
        })

    if iteration == 2:
        # Prioritize rows with harder questions, then sort by passage length descending
        row_details.sort(key=lambda x: (0 if x["is_hard"] else 1, -x["passage_len"]))
    else:
        # Prioritize rows with easier questions, then sort by passage length ascending
        row_details.sort(key=lambda x: (1 if x["is_hard"] else 0, x["passage_len"]))

    selected_rows = [x["row"] for x in row_details[:min(train_size, len(row_details))]]
    return Dataset.from_list(selected_rows)


def run_smoke_test(
    strategy: Strategy,
    train_dataset: Dataset,
    model: QwenInference,
) -> bool:
    """
    Run the strategy on up to 5 examples to verify structural validity.
    Returns True if it passes, False if it fails.

    TODO: Implement the pre-flight smoke test.
    Steps:
      1. Select up to 5 examples from train_dataset.
      2. Temporarily set model.max_new_tokens based on strategy.cot_format (4096 if CoT, 256 if direct).
      3. Evaluate the strategy on this subset using evaluate().
      4. Check that at least one predicted answer is not None (i.e. program extraction succeeded).
      5. Check that average output tokens generated per question does not exceed 90% of the token limit 
         (to prevent infinite looping/truncation).
      6. Return True if valid, False otherwise. Make sure to restore model.max_new_tokens at the end.
    """
    five_examples_dataset = train_dataset.shuffle(seed=42).select(range(min(5, len(train_dataset))))
    strategy_cot_format = strategy.cot_format
    original_max_tokens = model.max_new_tokens

    model.max_new_tokens = 4096 if strategy.cot_format != CoTFormat.NONE else 256

    print(f"Model max_new_tokens temporarily set to {model.max_new_tokens} for smoke test (strategy CoT format: {strategy_cot_format}).\n")
    evaluate_result = evaluate(strategy, split="smoke_test", dataset=five_examples_dataset, model=model)
    print(f"\nEvaluation results: {evaluate_result}")
    # Check that at least one predicted answer is not None
    has_valid_prediction = any(r.predicted_answer is not None for r in evaluate_result.per_question)
    # Check that average output tokens generated per question does not exceed 90% of the token limit
    avg_output_tokens = evaluate_result.total_output_tokens / len(evaluate_result.per_question) if evaluate_result.per_question else 0
    are_tokens_within_limit = avg_output_tokens <= 0.9 * model.max_new_tokens
    # Restore original max_new_tokens
    model.max_new_tokens = original_max_tokens
    return has_valid_prediction and are_tokens_within_limit


def run_evoagent(
    T: int,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    model: QwenInference,
    output_dir: Path,
    train_size: int = 100,
    resume_from: Optional[Path] = None,
    early_stop_accuracy: float = 1.0,
    afo_mode: str = "probabilistic",
    afo_prob_best: float = 0.4,
    afo_prob_original: float = 0.3,
    afo_prob_latest: float = 0.3,
    progressive_reflections: bool = True,
    use_curriculum: bool = False,
) -> StrategyHistory:
    """
    Run the EvoAgent loop for up to T iterations.

    TODO: Implement the EvoAgent optimization loop.
    For each iteration (from start_iteration up to T-1):
      1. Propose strategy:
         - Iteration 0: Use make_seed_strategy()
         - Iterations > 0: Mutate from a parent selected via select_parent_strategy().
           Try proposing up to 3 times, validating each using run_smoke_test().
      2. Set model.max_new_tokens dynamically (4096 if CoT, 256 if direct).
      3. Evaluate on train subset (curriculum or slice) and dev split.
      4. Accumulate token usage in TokenBudget.
      5. Reflect on errors (except on the last iteration T-1).
      6. Append/save strategies, evaluations, and reflections to the history file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.jsonl"
    history = StrategyHistory(history_path)

    if resume_from is not None:
        history.path = Path(resume_from)
        history.load()
        logger.info("Resumed from %s — %d strategies in history.", resume_from, len(history))
    elif history_path.exists():
        history.load()
        logger.info("Auto-resumed from %s — %d strategies in history.", history_path, len(history))

    budget = TokenBudget()
    start_iteration = len(history.strategies)

    if start_iteration >= T:
        logger.info("History already has %d strategies (T=%d). Nothing to do.", start_iteration, T)
        return history

    logger.info("Starting EvoAgent loop. Iterations %d–%d (T=%d).", start_iteration, T - 1, T)

    t0 = time.time()
    for iteration in tqdm(range(start_iteration, T)):
        parent_strategy = None
        meta_tokens_usage = 0
        # 1. Propose strategy:
        if iteration == 0:
            strategy = make_seed_strategy()
            print(f"\nIteration {iteration}: Using seed strategy.")
        else:
            parent_strategy = select_parent_strategy(history, 
                                                     afo_mode, 
                                                     afo_prob_best, 
                                                     afo_prob_original, 
                                                     afo_prob_latest)
            if parent_strategy is None:
                logger.warning("No parent strategy found for iteration %d. Using seed strategy.", iteration)
                strategy = make_seed_strategy()
            else:
                print(f"\nIteration {iteration}: Proposing new strategy from parent (ID: {parent_strategy.id}).")
                for attempt in range(3):
                    strategy, meta_tokens_usage = propose_self(history,
                                            model,
                                            max_retries=3,
                                            parent_strategy_id=parent_strategy.id,
                                            train_dataset=train_dataset)
                    if run_smoke_test(strategy, train_dataset, model):
                        print(f"    [PASS] Smoke test passed on attempt {attempt + 1}.")
                        break
                    else:
                        print(f"    [FAIL] Smoke test failed on attempt {attempt + 1}. Retrying...")
                else:
                    logger.error("Failed to propose a valid strategy after 3 attempts. Using parent strategy as fallback.")
                    strategy = parent_strategy
        
        # 2: Set model.max_new_tokens dynamically based on strategy.cot_format
        model.max_new_tokens = 4096 if strategy.cot_format != CoTFormat.NONE else 256
        
        # 3. Evaluate on train subset (curriculum or slice) and dev split.
        if use_curriculum:
            train_subset = select_curriculum_dataset(train_dataset, iteration, train_size)
        else: # Stratified sample: ensure all question types are represented
            _rng = _random.Random(42 + iteration)
            _by_type = collections.defaultdict(list)
            for _ex in train_dataset:
                _by_type[classify_question_type(_ex['answer'])].append(_ex)
            _per_type = max(1, train_size // len(_by_type))
            _stratified = []
            for _bucket in _by_type.values():
                _rng.shuffle(_bucket)
                _stratified.extend(_bucket[:_per_type])
            _rng.shuffle(_stratified)
            train_subset = Dataset.from_list(_stratified[:train_size])
        # Eval results for train and dev
        train_eval_result = evaluate(strategy, split="train", dataset=train_subset, model=model)
        dev_eval_result = evaluate(strategy, split="dev", dataset=dev_dataset, model=model)
        
        # 4. Accumulate token usage in TokenBudget.
        budget.add_eval(train_eval_result)
        budget.add_eval(dev_eval_result)
        budget.add_meta(meta_tokens_usage)
        
        # 5. Reflect on errors (except on the last iteration T-1).
        reflection = None
        reflection_tokens_usage = 0
        if iteration < T - 1:
            reflection, reflection_tokens_usage = reflect_self(strategy, train_eval_result, model, max_retries=5, progressive=progressive_reflections)
            budget.add_meta(reflection_tokens_usage)
        
        # Fill in StrategyMetadata for this iteration
        strategy.metadata = StrategyMetadata(
            dev_accuracy=dev_eval_result.accuracy,
            train_accuracy=train_eval_result.accuracy,
            parent_id=parent_strategy.id if iteration > 0 and parent_strategy is not None else None,
            iteration=iteration,
            token_cost_claude=meta_tokens_usage + reflection_tokens_usage,
            token_cost_qwen=(
                train_eval_result.total_input_tokens + train_eval_result.total_output_tokens +
                dev_eval_result.total_input_tokens + dev_eval_result.total_output_tokens
            ),
            extra={}
        )
        
        # 6. Append/save strategies, evaluations, and reflections to the history file.
        history.append_strategy(strategy)
        if reflection is not None:
            history.append_reflection(reflection)
            _save_reflection_json(reflection, output_dir, iteration)
        _save_strategy_json(strategy, output_dir, iteration)
        _save_eval_result(train_eval_result, output_dir, iteration, tag="train")
        _save_eval_result(dev_eval_result, output_dir, iteration, tag="dev")
        
        logger.info("Iteration %d completed in %.1fs", iteration, time.time() - t0)

        if dev_eval_result.accuracy >= early_stop_accuracy:
            logger.info("Early stop: dev accuracy %.3f reached threshold.", dev_eval_result.accuracy)
            # still save, then break
            break
    
    _print_leaderboard(history)
    logger.info("Token budget: %s", budget.summary())
    
    return history

# ------------------------------------------------------------------
# Internal Helpers for saving results
# ------------------------------------------------------------------

def _save_strategy_json(strategy: Strategy, output_dir: Path, iteration: int) -> None:
    path = output_dir / f"iter_{iteration:03d}_strategy.json"
    path.write_text(strategy.to_json(), encoding="utf-8")


def _save_eval_result(
    result: EvalResult,
    output_dir: Path,
    iteration: int,
    tag: str,
) -> None:
    import json
    path = output_dir / f"iter_{iteration:03d}_eval_{tag}.json"
    data = result.to_dict()
    data["per_question"] = [
        {
            "question_id": r.question_id,
            "question": r.question,
            "gold_answer": r.gold_answer,
            "gold_val": r.gold_val,
            "predicted_answer": r.predicted_answer,
            "predicted_val": r.predicted_val,
            "is_correct": r.is_correct,
            "question_type": r.question_type,
            "raw_output": r.raw_output,
        }
        for r in result.per_question
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_reflection_json(reflection, output_dir: Path, iteration: int) -> None:
    import json
    path = output_dir / f"iter_{iteration:03d}_reflection.json"
    path.write_text(
        json.dumps(reflection.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _print_leaderboard(history: StrategyHistory) -> None:
    rows = history.summary_table()
    if not rows:
        return
    header = f"{'Iter':>4}  {'ID':>8}  {'CoT':>10}  {'Dev Acc':>8}  {'Train Acc':>9}  {'Meta tok':>10}  {'Qwen tok':>8}"
    logger.info("Leaderboard:\n%s", header)
    for r in rows:
        dev = f"{r['dev_accuracy']:.3f}" if r["dev_accuracy"] is not None else "  —  "
        train = f"{r['train_accuracy']:.3f}" if r["train_accuracy"] is not None else "  —  "
        logger.info(
            "  %4d  %8s  %10s  %8s  %9s  %10d  %8d",
            r["iteration"],
            r["id"],
            r["cot_format"],
            dev,
            train,
            r["meta_tokens"],
            r["qwen_tokens"],
        )
