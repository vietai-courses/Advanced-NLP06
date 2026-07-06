"""
strategy.py — Strategy dataclass, enums, and StrategyHistory for EvoAgent.

A Strategy encodes a complete prompting approach: the template used to
construct prompts, the chain-of-thought format, few-shot examples, retrieval
configuration, and metadata about how it performed. StrategyHistory maintains
an ordered list of strategies and their associated reflections, and persists
everything to disk so runs are fully resumable.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CoTFormat(str, Enum):
    """How the model is instructed to reason before answering."""

    NONE = "none"
    STEPBYSTEP = "stepbystep"
    CHAIN = "chain"


@dataclass
class RetrievalConfig:
    """
    Configuration for optional retrieval-augmented prompting.

    enabled: whether to retrieve similar examples at inference time.
    top_k: number of retrieved examples to include.
    similarity_threshold: minimum cosine similarity to include an example.
    """

    enabled: bool = False
    top_k: int = 3
    similarity_threshold: float = 0.75

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievalConfig":
        return cls(**d)


@dataclass
class StrategyMetadata:
    """
    Runtime metadata for a strategy: performance scores, lineage, and cost.

    dev_accuracy: accuracy on the dev split after evaluation.
    train_accuracy: accuracy on the training subset used for evaluation.
    parent_id: id of the strategy this was proposed from (None for seed).
    iteration: which EvoAgent iteration produced this strategy.
    token_cost_claude: total meta-agent tokens consumed proposing + reflecting.
        Field name is kept for compatibility with existing histories.
    token_cost_qwen: total Qwen tokens consumed evaluating this strategy.
    extra: arbitrary key-value pairs for extensibility.
    """

    dev_accuracy: Optional[float] = None
    train_accuracy: Optional[float] = None
    parent_id: Optional[str] = None
    iteration: int = 0
    token_cost_claude: int = 0
    token_cost_qwen: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyMetadata":
        return cls(**d)


@dataclass
class FewShotExample:
    """A single few-shot example packed into a strategy."""

    passage: str
    question: str
    answer: str              # "A", "B", "C", or "D" (or program)
    reasoning: Optional[str] = None  # Optional chain-of-thought for the example

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FewShotExample":
        # Drop legacy "choices" field if present from old history files
        if "choices" in d:
            del d["choices"]
        return cls(**d)


@dataclass
class Strategy:
    """
    A complete prompting strategy for the Financial QA task.

    id: unique identifier (UUID4 string).
    prompt_template: a Python format string with placeholders:
        {passage}, {question}, {few_shot_block}, {cot_instruction}.
    cot_format: how to elicit chain-of-thought reasoning.
    few_shot_examples: in-context examples prepended to the prompt.
    retrieval_config: optional retrieval-augmented generation settings.
    metadata: performance scores, lineage, and token costs.
    """

    id: str
    prompt_template: str
    cot_format: CoTFormat
    few_shot_examples: list[FewShotExample]
    retrieval_config: RetrievalConfig
    metadata: StrategyMetadata

    def __post_init__(self) -> None:
        """Automatically escape any invalid formatting curly braces."""
        import re
        # Normalize spacing inside valid placeholders
        template = self.prompt_template
        template = re.sub(
            r"\{\s*(passage|question|few_shot_block|cot_instruction)\s*\}",
            lambda m: f"{{{m.group(1)}}}",
            template,
        )
        
        # 1. Replace double braces with placeholders to protect them
        temp_double_open = "__TEMP_DOUBLE_OPEN_BRACE_PLACEHOLDER__"
        temp_double_close = "__TEMP_DOUBLE_CLOSE_BRACE_PLACEHOLDER__"
        cleaned = template.replace("{{", temp_double_open).replace("}}", temp_double_close)
        
        # 2. Replace valid placeholders with temporary tokens
        valid_placeholders = {
            "{passage}": "__VALID_PASSAGE_PLACEHOLDER__",
            "{question}": "__VALID_QUESTION_PLACEHOLDER__",
            "{few_shot_block}": "__VALID_FEW_SHOT_PLACEHOLDER__",
            "{cot_instruction}": "__VALID_COT_PLACEHOLDER__"
        }
        
        for ph, token in valid_placeholders.items():
            cleaned = cleaned.replace(ph, token)
            
        # 3. Escape all remaining single curly braces
        cleaned = cleaned.replace("{", "{{").replace("}", "}}")
        
        # 4. Restore valid placeholders
        for ph, token in valid_placeholders.items():
            cleaned = cleaned.replace(token, ph)
            
        # 5. Restore original double braces
        cleaned = cleaned.replace(temp_double_open, "{{").replace(temp_double_close, "}}")
        
        self.prompt_template = cleaned

    def to_dict(self) -> dict:
        """Serialize the strategy to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "prompt_template": self.prompt_template,
            "cot_format": self.cot_format.value,
            "few_shot_examples": [ex.to_dict() for ex in self.few_shot_examples],
            "retrieval_config": self.retrieval_config.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Strategy":
        """Deserialize a strategy from a dictionary."""
        return cls(
            id=d["id"],
            prompt_template=d["prompt_template"],
            cot_format=CoTFormat(d["cot_format"]),
            few_shot_examples=[FewShotExample.from_dict(ex) for ex in d["few_shot_examples"]],
            retrieval_config=RetrievalConfig.from_dict(d["retrieval_config"]),
            metadata=StrategyMetadata.from_dict(d["metadata"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Strategy":
        return cls.from_dict(json.loads(s))


@dataclass
class Reflection:
    """
    Structured analysis of a strategy's performance produced by the reflector.

    strategy_id: which strategy this reflects on.
    accuracy_by_type: mapping from question category to accuracy (0-1).
    top_failures: list of dicts describing the worst failure cases.
    hypothesis: a concrete, testable hypothesis for why the strategy failed
        in certain categories and what the next strategy should try.
    summary: one-paragraph prose summary of the reflection.
    raw_response: the full meta-agent response, kept for debugging.
    """

    strategy_id: str
    accuracy_by_type: dict[str, float]
    top_failures: list[dict[str, Any]]
    hypothesis: str
    summary: str
    raw_response: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Reflection":
        return cls(**d)


class StrategyHistory:
    """
    Ordered list of (Strategy, optional Reflection) pairs.

    Persists to a JSONL file so every iteration is appended atomically.
    This means an interrupted run can be resumed from the last complete
    iteration without re-running earlier strategies.

    File format: one JSON object per line, alternating strategy / reflection
    records tagged with a "record_type" field.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        # Parallel lists: strategies[i] and reflections[i] are matched.
        self.strategies: list[Strategy] = []
        self.reflections: list[Optional[Reflection]] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all records from disk. Safe to call on a new (empty) file."""
        if not self.path.exists():
            logger.info("No history file found at %s — starting fresh.", self.path)
            return

        logger.info("Loading strategy history from %s", self.path)
        pending_strategy: Optional[Strategy] = None

        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line %d: %s", lineno, exc)
                    continue

                record_type = record.get("record_type")
                if record_type == "strategy":
                    # If there's a pending strategy without a reflection, flush it.
                    if pending_strategy is not None:
                        self.strategies.append(pending_strategy)
                        self.reflections.append(None)
                    pending_strategy = Strategy.from_dict(record["data"])

                elif record_type == "reflection":
                    if pending_strategy is None:
                        logger.warning("Reflection at line %d has no preceding strategy — skipping.", lineno)
                        continue
                    self.strategies.append(pending_strategy)
                    self.reflections.append(Reflection.from_dict(record["data"]))
                    pending_strategy = None

                else:
                    logger.warning("Unknown record_type '%s' at line %d — skipping.", record_type, lineno)

        # Flush any trailing strategy without a reflection (run was interrupted
        # before reflection completed).
        if pending_strategy is not None:
            self.strategies.append(pending_strategy)
            self.reflections.append(None)

        logger.info("Loaded %d strategies from history.", len(self.strategies))

    def append_strategy(self, strategy: Strategy) -> None:
        """Append a strategy record. Does not pair with a reflection yet."""
        self.strategies.append(strategy)
        self.reflections.append(None)
        self._write_record("strategy", strategy.to_dict())
        logger.debug("Appended strategy %s to history.", strategy.id)

    def append_reflection(self, reflection: Reflection) -> None:
        """
        Pair the most recently appended strategy with its reflection.
        Raises ValueError if the last strategy already has a reflection.
        """
        if not self.strategies:
            raise ValueError("No strategies in history to attach a reflection to.")
        last_idx = len(self.strategies) - 1
        if self.reflections[last_idx] is not None:
            raise ValueError(
                f"Strategy {self.strategies[last_idx].id} already has a reflection."
            )
        self.reflections[last_idx] = reflection
        self._write_record("reflection", reflection.to_dict())
        logger.debug("Appended reflection for strategy %s.", reflection.strategy_id)

    def update_strategy_metadata(self, strategy_id: str, metadata: StrategyMetadata) -> None:
        """
        Update the metadata for a strategy already in memory.
        Rewrites the entire history file to persist the change.
        """
        for i, s in enumerate(self.strategies):
            if s.id == strategy_id:
                self.strategies[i].metadata = metadata
                break
        else:
            raise ValueError(f"Strategy {strategy_id} not found in history.")
        self._rewrite()

    def _write_record(self, record_type: str, data: dict) -> None:
        """Atomically append a single JSONL record to the history file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = json.dumps({"record_type": record_type, "data": data}, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record + "\n")

    def _rewrite(self) -> None:
        """Rewrite the entire file from in-memory state (used after metadata updates)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for strategy, reflection in zip(self.strategies, self.reflections):
                record = json.dumps(
                    {"record_type": "strategy", "data": strategy.to_dict()},
                    ensure_ascii=False,
                )
                fh.write(record + "\n")
                if reflection is not None:
                    record = json.dumps(
                        {"record_type": "reflection", "data": reflection.to_dict()},
                        ensure_ascii=False,
                    )
                    fh.write(record + "\n")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.strategies)

    def latest_strategy(self) -> Optional[Strategy]:
        """Return the most recently appended strategy, or None."""
        return self.strategies[-1] if self.strategies else None

    def latest_reflection(self) -> Optional[Reflection]:
        """Return the most recently stored reflection, or None."""
        for r in reversed(self.reflections):
            if r is not None:
                return r
        return None

    def best_strategy(self) -> Optional[Strategy]:
        """Return the strategy with the highest dev_accuracy, or None."""
        scored = [s for s in self.strategies if s.metadata.dev_accuracy is not None]
        if not scored:
            return None
        return max(scored, key=lambda s: s.metadata.dev_accuracy)  # type: ignore[return-value]

    def summary_table(self) -> list[dict]:
        """Return a list of dicts suitable for printing or logging."""
        rows = []
        for s, r in zip(self.strategies, self.reflections):
            rows.append(
                {
                    "id": s.id[:8],
                    "iteration": s.metadata.iteration,
                    "cot_format": s.cot_format.value,
                    "dev_accuracy": s.metadata.dev_accuracy,
                    "train_accuracy": s.metadata.train_accuracy,
                    "meta_tokens": s.metadata.token_cost_claude,
                    "qwen_tokens": s.metadata.token_cost_qwen,
                    "has_reflection": r is not None,
                }
            )
        return rows


def make_seed_strategy() -> Strategy:
    """
    Build the iteration-0 baseline strategy: program generation prompt.
    """
    template = (
        "Bạn hãy đóng vai trò là một chuyên gia phân tích tài chính. "
        "Nhiệm vụ của bạn là giải bài toán tài chính sau đây bằng cách lập chương trình (hàm toán học).\n\n"
        "{few_shot_block}"
        "Bối cảnh:\n{passage}\n\n"
        "Câu hỏi: {question}"
    )
    ex1 = FewShotExample(
        passage="kqkd năm 2020:doanh thu và lợi nhuận sau thuế (lnst) lần lượt đạt 11.228 tỷ đồng (-40% yoy) và 74 tỷ đồng (-82% yoy), theo đó chúng tôi ước tính eps là 290đồng/cổ phiếu (-83% yoy). cho năm 2021: chúng tôi dự phóng doanh thu, lnst và eps lần lượt là 11.565 tỷ đồng (+3% yoy), 85 tỷ đồng (+13,5% yoy) và 330 đồng/cổ phiếu (+13,5% yoy).",
        question="Hãy tính doanh thu dự phóng cho năm 2021 dựa trên số liệu doanh thu năm 2020.",
        answer="multiply(11228, 1.03)",
        reasoning="Doanh thu năm 2020 là 11.228 tỷ đồng và doanh thu năm 2021 dự phóng tăng trưởng 3% so với năm 2020. Do đó, doanh thu dự phóng 2021 = 11228 * (1 + 0.03) = 11228 * 1.03."
    )
    ex2 = FewShotExample(
        passage="tỷ suất cổ tức của ht1 năm 2022 và năm 2023 dự phóng lần lượt ở mức 3.6% và 7.1%, tương ứng với cổ tức tiền mặt năm 2022 & 2023 lần lượt là vnd 600 đồng/cp và vnd1,200 đồng/cp.",
        question="Tổng cổ tức tiền mặt trên mỗi cổ phiếu cho năm 2022 và 2023 là bao nhiêu (VND)?",
        answer="add(600, 1200)",
        reasoning="Cổ tức tiền mặt dự phóng năm 2022 là 600 đồng/cp và năm 2023 là 1.200 đồng/cp. Tổng cổ tức 2 năm = 600 + 1200."
    )
    ex3 = FewShotExample(
        passage="lhg ghi nhận doanh thu (tỷ đồng) qua các năm như sau: 2017: 781, 2018: 820, 2019: 650, 2020: 590, 2021: 710, 2022(F): 850. bảng trên cho thấy biến động doanh thu của lhg trong giai đoạn này.",
        question="Doanh thu trung bình hàng năm của LHG trong giai đoạn 2017-2022(F) là bao nhiêu tỷ đồng?",
        answer="table_average(Doanh thu, none)",
        reasoning="Câu hỏi hỏi TRUNG BÌNH của toàn bộ cột 'Doanh thu' qua các năm. Dùng table_average(Doanh thu, none) — không tự cộng từng ô."
    )
    return Strategy(
        id=str(uuid.uuid4()),
        prompt_template=template,
        cot_format=CoTFormat.NONE,
        few_shot_examples=[ex1, ex2, ex3],
        retrieval_config=RetrievalConfig(enabled=False),
        metadata=StrategyMetadata(iteration=0),
    )

