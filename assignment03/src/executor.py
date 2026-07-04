"""
executor.py — Evaluate a Strategy on a dataset split.

The executor translates a Strategy into per-example prompts, runs batched
inference with QwenInference, collects predicted vs. gold answers, and returns
an EvalResult with accuracy broken down by question type (if available).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from datasets import Dataset
from tqdm import tqdm

from src.model import QwenInference, extract_answer
from src.strategy import CoTFormat, FewShotExample, Strategy

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Per-question prediction record stored for the reflector."""

    question_id: str           # Row index as string (or dataset-provided id)
    passage: str
    question: str

    gold_answer: str           # Gold math program (e.g. "subtract(100, 50)")
    predicted_answer: Optional[str]  # None if extraction failed
    is_correct: bool
    raw_output: str
    question_type: str = "unknown"  # Heuristic category (addition, division, etc.)
    input_tokens: int = 0
    output_tokens: int = 0
    gold_val: Optional[float] = None
    predicted_val: Optional[float] = None


@dataclass
class EvalResult:
    """Aggregate evaluation results for one strategy on one dataset split."""

    strategy_id: str
    split: str
    num_examples: int
    num_correct: int
    accuracy: float
    accuracy_by_type: dict[str, float] = field(default_factory=dict)
    count_by_type: dict[str, int] = field(default_factory=dict)
    per_question: list[QuestionResult] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "split": self.split,
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "accuracy": self.accuracy,
            "accuracy_by_type": self.accuracy_by_type,
            "count_by_type": self.count_by_type,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def failures(self, top_k: int = 10) -> list[QuestionResult]:
        """Return up to top_k incorrect predictions."""
        return [r for r in self.per_question if not r.is_correct][:top_k]


# ------------------------------------------------------------------
# Token Budget Vault
# ------------------------------------------------------------------

@dataclass
class TokenBudget:
    """Running totals for token usage across the full run."""

    qwen_input: int = 0
    qwen_output: int = 0
    meta_total: int = 0

    def add_eval(self, result: EvalResult) -> None:
        """
        TODO: Implement this method to accumulate token usage from an EvalResult
        into qwen_input and qwen_output.
        """
        self.qwen_input += result.total_input_tokens
        self.qwen_output += result.total_output_tokens

    def add_meta(self, tokens: int) -> None:
        """
        TODO: Implement this method to accumulate meta-agent token usage (reflect/propose).
        """
        self.meta_total += tokens

    @property
    def qwen_total(self) -> int:
        """
        TODO: Return the sum of qwen_input and qwen_output tokens.
        """
        return self.qwen_input + self.qwen_output

    def summary(self) -> str:
        return (
            f"Token usage — Qwen: {self.qwen_total:,} (in={self.qwen_input:,}, "
            f"out={self.qwen_output:,}) | Meta-agent: {self.meta_total:,}"
        )


# ------------------------------------------------------------------
# Question-type heuristics
# ------------------------------------------------------------------

def normalize_program(prog: Optional[str]) -> str:
    """Normalize program string by stripping all whitespace and lowercasing."""
    if not prog:
        return ""
    return "".join(prog.split()).lower()


def classify_question_type(program: str) -> str:
    """
    Assign a question type based on the operations in the program.

    Returns one of: addition, subtraction, multiplication, division, table_op, or "other".
    """
    if not program:
        return "other"
    p_lower = program.lower()
    if "add" in p_lower:
        return "addition"
    elif "subtract" in p_lower:
        return "subtraction"
    elif "multiply" in p_lower:
        return "multiplication"
    elif "divide" in p_lower:
        return "division"
    elif "table_" in p_lower:
        return "table_op"
    return "other"


# ------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------

_COT_INSTRUCTIONS: dict[CoTFormat, str] = {
    CoTFormat.NONE: "",
    CoTFormat.STEPBYSTEP: (
        "Hãy suy nghĩ từng bước trước khi đưa ra đáp án. "
        "Cuối cùng, viết chương trình dưới dạng 'Chương trình: <chương_trình>'."
    ),
    CoTFormat.CHAIN: (
        "Hãy lập luận theo chuỗi suy nghĩ (chain-of-thought) để giải thích cách giải. "
        "Kết thúc bằng 'Chương trình: <chương_trình>'."
    ),
}

_SYSTEM_MESSAGE = (
    "Bạn là một trợ lý AI chuyên phân tích tài chính tiếng Việt. "
    "Nhiệm vụ của bạn là viết chương trình dạng các hàm toán học để trả lời câu hỏi dựa trên văn bản và bảng số liệu được cung cấp."
)

_DSL_BLOCK = """QUY TẮC VIẾT CHƯƠNG TRÌNH:
1. Mỗi bước là một hàm riêng biệt, phân cách bằng dấu phẩy
2. KHÔNG lồng hàm vào nhau (không dùng divide(table_average(...), ...))
3. Dùng #0, #1, #2... để tham chiếu kết quả của bước trước (bắt đầu từ #0)
4. Tên cột/hàng trong table_xxx KHÔNG dùng dấu ngoặc kép
5. table_xxx chỉ nhận đúng 2 tham số: (tên_hàng, none)
6. Số âm viết trực tiếp: add(-167.4, -53.3) — không dùng ngoặc thêm
7. CỰC KỲ QUAN TRỌNG VỀ TỶ LỆ PHẦN TRĂM: Kết quả đầu ra của chương trình PHẢI luôn ở dạng tỷ lệ thập phân (ví dụ: 0.05 thay vì 5%, hay 0.03124 thay vì 3.124%). Tuyệt đối KHÔNG nhân thêm 100 ở bước cuối cùng của chương trình (KHÔNG dùng multiply(#X, 100) cho các câu hỏi tính phần trăm).
8. CỰC KỲ QUAN TRỌNG: Nếu cần một giá trị cụ thể từ bảng (ví dụ: doanh thu năm 2022), KHÔNG dùng hàm table_xxx. Hãy tự đọc bảng và viết TRỰC TIẾP con số đó vào hàm toán học.
9. CỰC KỲ QUAN TRỌNG: Nếu câu hỏi yêu cầu tính chênh lệch hoặc so sánh đơn thuần mà không có từ 'phần trăm' hoặc '%', chỉ sử dụng duy nhất phép trừ (subtract) — KHÔNG tự động thêm bước chia (divide) để tính tỷ lệ.

Ví dụ đúng:
  subtract(108.50, 100), divide(#0, 100) (Trích xuất trực tiếp số 108.50 và 100 từ văn bản/bảng)
  table_max(Lãi ròng, none), table_min(Lãi ròng, none), subtract(#0, #1) (Tính max/min trên toàn bộ hàng)
  divide(115.18, 100), divide(113.68, 100), subtract(#0, 1), subtract(#1, 1), add(#2, #3), divide(#4, 2)
"""


def build_few_shot_block(examples: list[FewShotExample]) -> str:
    """
    Format a list of few-shot examples into a single string block.
    """
    if not examples:
        return ""

    parts = ["Dưới đây là một số ví dụ mẫu:\n"]
    for idx, ex in enumerate(examples, start=1):
        is_full_cot = ex.reasoning and "</think>" in ex.reasoning
        if is_full_cot:
            parts.append(
                f"Ví dụ {idx}:\n"
                f"Đoạn văn: {ex.passage}\n"
                f"Câu hỏi: {ex.question}\n"
                f"Câu trả lời:\n{ex.reasoning}\n"
            )
        else:
            reasoning_part = f"\nGiải thích: {ex.reasoning}" if ex.reasoning else ""
            parts.append(
                f"Ví dụ {idx}:\n"
                f"Đoạn văn: {ex.passage}\n"
                f"Câu hỏi: {ex.question}\n"
                f"Chương trình: {ex.answer}{reasoning_part}\n"
            )
    parts.append("---\nBây giờ hãy giải câu hỏi dưới đây:\n")
    return "\n".join(parts)


def build_prompt(
    strategy: Strategy,
    passage: str,
    question: str,
) -> str:
    """
    Construct the user-facing portion of the prompt for one example.
    """
    few_shot_block = build_few_shot_block(strategy.few_shot_examples)

    # Check if this is a legacy template containing placeholders
    if "{passage}" in strategy.prompt_template:
        try:
            user_message = strategy.prompt_template.format(
                passage=passage,
                question=question,
                few_shot_block=few_shot_block,
                cot_instruction="",
            )
        except KeyError as exc:
            raise ValueError(
                f"Strategy {strategy.id} prompt_template references unknown key {exc}. "
                "Valid keys: passage, question, few_shot_block, cot_instruction."
            ) from exc
    else:
        # Structured scaffolding: General phrasing -> few-shot -> passage -> question
        parts = []
        if strategy.prompt_template:
            parts.append(strategy.prompt_template.strip())
        if few_shot_block:
            parts.append(few_shot_block.strip())
        
        parts.append(f"Bối cảnh:\n{passage}")
        parts.append(f"Câu hỏi: {question}")
        user_message = "\n\n".join(parts)

    # Inject DSL rules unconditionally
    user_message = f"{user_message.strip()}\n\n{_DSL_BLOCK}"
    
    # Inject CoT and output format anchor
    if strategy.cot_format != CoTFormat.NONE:
        user_message = (
            f"{user_message}\n"
            f"Quy tắc đầu ra cho mô hình lập luận (CoT):\n"
            f"1. Hãy trình bày quá trình lập luận và suy nghĩ thật ngắn gọn bên trong cặp thẻ <think>...</think> (tối đa 200 từ, tập trung hoàn toàn vào công thức toán học và các bước tính toán, KHÔNG viết dông dài hay lặp lại câu hỏi).\n"
            f"2. Ngay sau thẻ đóng </think>, hãy trả về một khối JSON duy nhất chứa câu trả lời với cấu trúc và 3 khóa chính xác như sau (không thêm bất kỳ văn bản nào bên ngoài khối JSON này):\n"
            f"{{\n"
            f"  \"Reasoning\": \"Giải thích ngắn gọn bằng tiếng Việt (tối đa 2-3 câu) về cách tính toán\",\n"
            f"  \"Program syntax\": \"Chương trình toán học dạng phẳng viết theo DSL (ví dụ: subtract(7.758, 7.523), divide(#0, 7.523))\",\n"
            f"  \"Numerical result\": \"Kết quả số học tính toán cuối cùng ở dạng số hoặc tỷ lệ thập phân (ví dụ: 0.03124)\"\n"
            f"}}\n"
            f"Chú ý: KHÔNG đặt khối JSON bên trong bất kỳ thẻ block code markdown nào khác, chỉ trả về khối JSON trần ngay sau </think>."
        )
    else:
        user_message = (
            f"{user_message}\n"
            f"Chỉ trả về chương trình duy nhất trên một dòng bắt đầu bằng `PROGRAM:` và không thêm bất kỳ từ ngữ nào khác.\n"
            f"Ví dụ: PROGRAM: subtract(108.50, 100), divide(#0, 100)"
        )

    return user_message


# ------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------

def evaluate(
    strategy: Strategy,
    split: str,
    dataset: Dataset,
    model: QwenInference,
) -> EvalResult:
    """
    Evaluate a strategy on a HuggingFace Dataset split.

    TODO: Implement this function.
    Steps:
      1. Parse each row in dataset (using _parse_row) to get context/questions.
      2. Call build_prompt() to build prompts, and format them using model.format_prompt().
      3. Run batched inference using model.generate_batch(prompts, cot_format=...).
      4. Compare each prediction to gold program:
         - Evaluate parsed program using evaluate_program(predicted_ans, table).
         - Floating-point accuracy: compare program execution values (abs diff <= 1e-4).
         - Fallback: check normalized exact program string matches.
      5. Aggregate counts: total_correct, total input/output tokens, accuracy, 
         accuracy by question type, count by type.
      6. Return EvalResult.
    """
    from src.evaluator import evaluate_program

    logger.info("Evaluating strategy %s on %s split.", strategy.id[:8], split)
    start_time = time.time()

    correct_count_by_type : dict[str, int] = {}
    total_correct : int = 0
    total_input_tokens : int = 0
    total_output_tokens : int = 0
    accuracy : float = 0.0
    accuracy_by_type : dict[str, float] = {}
    count_by_type : dict[str, int] = {}
    per_question = []
    formatted_prompts = []
    parsed_dataset_rows = []
    
    for idx, row in enumerate(dataset):
        passage, question, gold_program, q_type, table, exe_ans = _parse_row(row, idx)[0]
        count_by_type[q_type] = count_by_type.get(q_type, 0) + 1
        parsed_dataset_rows.append((passage, question, gold_program, q_type, table, exe_ans))
        prompt = build_prompt(strategy, passage, question)
        formatted_prompt = model.format_prompt(system_message=_SYSTEM_MESSAGE, user_message=prompt, enable_thinking=False)
        formatted_prompts.append(formatted_prompt)

    generation_result = model.generate_batch(formatted_prompts, cot_format=False)

    for idx, (passage, question, gold_program, q_type, table, exe_ans) in enumerate(parsed_dataset_rows):
        predicted_ans = generation_result[idx].predicted_answer
        is_correct = False
        try:
            evaluate_result = evaluate_program(predicted_ans, table)
            is_correct = abs(evaluate_result - float(exe_ans)) <= 1e-4
        except Exception as e:
            print(f"Exception {e} | eval_result: {evaluate_result} | exe_ans: {exe_ans} | predicted_ans: {predicted_ans}")
            is_correct = (predicted_ans == gold_program)

        per_question_result = QuestionResult(
            question_id=str(idx),
            passage=passage,
            question=question,
            gold_answer=gold_program,
            predicted_answer=predicted_ans,
            is_correct=is_correct,
            raw_output=generation_result[idx].raw_output,
            question_type=q_type,
            input_tokens=generation_result[idx].input_tokens,
            output_tokens=generation_result[idx].output_tokens,
            gold_val=float(exe_ans),
            predicted_val=evaluate_result
        )
        per_question.append(per_question_result)
        total_correct += 1 if is_correct else 0
        total_input_tokens += generation_result[idx].input_tokens
        total_output_tokens += generation_result[idx].output_tokens
        if is_correct:
            correct_count_by_type[q_type] = correct_count_by_type.get(q_type, 0) + 1
    
    # Get overall accuracy and by type accuracy
    accuracy = total_correct / len(dataset) if len(dataset) > 0 else 0.0
    for q_type, count in count_by_type.items():
        correct_count = correct_count_by_type.get(q_type, 0)
        accuracy_by_type[q_type] = correct_count / count if count > 0 else 0.0

    return EvalResult(
        strategy_id=strategy.id,
        split=split,
        num_examples=len(dataset),
        num_correct=total_correct,
        accuracy=accuracy,
        accuracy_by_type=accuracy_by_type,
        count_by_type=count_by_type,
        per_question=per_question,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        elapsed_seconds=time.time() - start_time
    )

# ------------------------------------------------------------------
# Row parsing helper
# ------------------------------------------------------------------

def _parse_row(row: dict, idx: int) -> list[tuple[str, str, str, str, list[list[str]], str]]:
    """
    Parse a flat dataset row into a single evaluation tuple (passage, question, gold, q_type, table, exe_ans).
    """
    passage = row.get("context") or ""
    question = row.get("question") or ""
    gold = row.get("answer") or ""
    table = row.get("table") or []
    exe_ans = row.get("exe_ans") or ""
    
    # Classify based on the gold program operations
    q_type = classify_question_type(gold)

    # Return as a single question result
    return [(passage, question, gold, q_type, table, exe_ans)]
