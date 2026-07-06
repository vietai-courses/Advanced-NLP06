"""
self_proposer.py — Self-optimization proposer: uses Qwen itself as the meta-agent.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Optional, List
from pydantic import BaseModel, Field
from datasets import Dataset

from src.model import extract_answer
from src.executor import normalize_program, classify_question_type
from src.strategy import (
    CoTFormat,
    FewShotExample,
    RetrievalConfig,
    Strategy,
    StrategyHistory,
    StrategyMetadata,
)

class FewShotExampleSchema(BaseModel):
    passage: str
    question: str
    answer: str
    reasoning: Optional[str] = None

class ProposerSchema(BaseModel):
    hypothesis: str = Field(..., description="A short one-sentence hypothesis")
    instruction_phrasing: str = Field(..., description="General instructions / role / phrasing prefix for the model, without any placeholders")
    cot_format: str = Field(..., description="Must be 'none', 'stepbystep', or 'chain'")
    few_shot_examples: List[FewShotExampleSchema]
    reasoning: str = Field(..., description="A short one-sentence reasoning")

logger = logging.getLogger(__name__)

_VALID_COT = {f.value for f in CoTFormat}

_SYSTEM_PROPOSE = """\
Bạn là trợ lý nghiên cứu NLP đang thiết kế một chiến lược prompting \
để giúp một mô hình ngôn ngữ giải bài tập toán tài chính tiếng Việt (cộng, trừ, nhân, chia, đọc bảng).

Nhiệm vụ của bạn là đưa ra một chiến lược prompting mới dựa trên lịch sử các chiến lược đã thử và kết quả phản ánh gần nhất.

LƯU Ý QUAN TRỌNG VỀ CÚ PHÁP CHƯƠNG TRÌNH (BẮT BUỘC TUÂN THỦ TRONG FEW-SHOT EXAMPLES):
1. Mỗi bước là một hàm riêng biệt, phân cách bằng dấu phẩy
2. KHÔNG lồng hàm vào nhau (không dùng divide(table_average(...), ...))
3. Dùng #0, #1, #2... để tham chiếu kết quả của bước trước (bắt đầu từ #0)
4. Tên cột/hàng trong table_xxx KHÔNG dùng dấu ngoặc kép
5. table_xxx chỉ nhận đúng 2 tham số: (tên_hàng, none)
6. Số âm viết trực tiếp: add(-167.4, -53.3) — không dùng ngoặc thêm
7. CỰC KỲ QUAN TRỌNG VỀ TỶ LỆ PHẦN TRĂM: Kết quả đầu ra của chương trình PHẢI luôn ở dạng tỷ lệ thập phân (ví dụ: 0.05 thay vì 5%, hay 0.03124 thay vì 3.124%). Tuyệt đối KHÔNG nhân thêm 100 ở bước cuối cùng của chương trình (KHÔNG dùng multiply(#X, 100) cho các câu hỏi tính phần trăm).
8. Nếu cần giá trị một ô CỤ THỂ từ bảng (ví dụ: doanh thu năm 2022 = 500 tỷ), đọc trực tiếp và viết số đó vào hàm, KHÔNG dùng table_xxx.
9. CỰC KỲ QUAN TRỌNG — table_max / table_min / table_average / table_sum: Khi câu hỏi hỏi GIÁ TRỊ LỚN NHẤT / NHỎ NHẤT / TRUNG BÌNH / TỔNG của CẢ MỘT CỘT hoặc HÀNG trong bảng, BẮT BUỘC dùng table_max / table_min / table_average / table_sum. TUYỆT ĐỐI KHÔNG tự cộng/trừ từng ô thay thế.
10. CỰC KỲ QUAN TRỌNG: Nếu câu hỏi yêu cầu tính chênh lệch hoặc so sánh đơn thuần mà không có từ 'phần trăm' hoặc '%', chỉ sử dụng duy nhất phép trừ (subtract) — KHÔNG tự động thêm bước chia (divide) để tính tỷ lệ.

Ví dụ đúng:
  subtract(7.758, 7.523), divide(#0, 7.523) (Tính tỷ lệ tăng trưởng phần trăm dưới dạng tỷ lệ thập phân, không nhân 100)
  divide(99782, 2626154) (Tính phần trăm dưới dạng thập phân, không nhân 100)
  multiply(11228, 1.03) (Nhân trực tiếp giá trị tăng trưởng 3% dự phóng)
  table_max(Lãi ròng, none), table_min(Lãi ròng, none), subtract(#0, #1) (Tính max/min trên toàn bộ hàng)

LƯU Ý QUAN TRỌNG VỀ ĐỊNH DẠNG CHIẾN LƯỢC:
- Định dạng suy luận (cot_format) có thể chọn từ: "none" (không suy nghĩ trước khi trả lời, direct program), "stepbystep" (suy nghĩ từng bước ngắn gọn), hoặc "chain" (lập luận đầy đủ).
- Trích xuất 1-2 ví dụ few-shot từ Failure Logs (giữ ngắn gọn). Các ví dụ few-shot phải viết theo đúng Cú Pháp Chương Trình ở trên.
- instruction_phrasing là phần hướng dẫn/phong cách/vai trò chung viết bằng tiếng Việt. KHÔNG chứa các chuỗi giữ chỗ như {passage}, {question}, {few_shot_block} vì hệ thống tự động chèn.

YÊU CẦU ĐỘ DÀI VÀ CẤU TRÚC (BẮT BUỘC):
- Viết ngắn gọn và súc tích. Tổng độ dài toàn bộ câu trả lời KHÔNG ĐƯỢC vượt quá 500 từ.
- KHÔNG lặp từ, không giải thích dông dài, không tự tạo ra văn bản rác hoặc ký tự lặp vô nghĩa.
"""


def _is_valid_dsl_program(program: str) -> bool:
    """
    Validate that the program syntax matches the FinQA DSL constraints.
    - Must not contain '=', '+', '*', or '/' (which indicate raw arithmetic equations, not DSL functions).
    - Can contain minus sign '-' if it represents a negative number (e.g., add(-167.4, -53.3)).
    - Must contain at least one valid DSL operator or a step reference (e.g. #0).

    TODO: Implement this validation check.
    """
    list_of_invalid_chars = ['=', '+', '*', '/']
    if any(char in program for char in list_of_invalid_chars):
        print(f"Program cannot contain =, +, *, /: {program}")
        return False
    if "-" in program and not re.search(r'(?<!\d)-|-(?!\d)', program):
        print(f"Program cannot contain '-' unless it's part of a negative number: {program}")
        return False
    dsl_q_type = classify_question_type(program)
    if dsl_q_type == "other":
        # It might be a step reference like #0
        if not re.search(r'#\d+', program):
            print(f"Program must contain at least one valid DSL operator or a step reference: {program}")
            return False
    return True


def _build_propose_message(history: StrategyHistory, parent_strategy_id: Optional[str] = None) -> str:
    lines = ["=== Lịch sử chiến lược ==="]
    for s, r in zip(history.strategies, history.reflections):
        acc = f"{s.metadata.dev_accuracy:.3f}" if s.metadata.dev_accuracy is not None else "chưa đánh giá"
        lines.append(
            f"\nIteration {s.metadata.iteration} | ID: {s.id[:8]} | dev_accuracy={acc} | cot={s.cot_format.value}"
        )
        lines.append(f"  Template: {s.prompt_template[:300]!r}")
        if r is not None:
            lines.append(f"  Loại câu hỏi yếu nhất: {min(r.accuracy_by_type, key=r.accuracy_by_type.get) if r.accuracy_by_type else 'unknown'}")
            lines.append(f"  Giả thuyết: {r.hypothesis[:200]}")

    # Find parent strategy
    parent_strategy = None
    parent_reflection = None
    if parent_strategy_id is not None:
        for s, r in zip(history.strategies, history.reflections):
            if s.id == parent_strategy_id:
                parent_strategy = s
                parent_reflection = r
                break
    if parent_strategy is None:
        parent_strategy = history.latest_strategy()
        parent_reflection = history.latest_reflection()

    if parent_strategy is not None:
        lines.append("\n=== Chiến lược gốc cần tối ưu (Parent Strategy) ===")
        lines.append(f"ID: {parent_strategy.id[:8]}")
        lines.append(f"CoT: {parent_strategy.cot_format.value}")
        lines.append(f"Template:\n{parent_strategy.prompt_template}")
        if parent_reflection is not None:
            lines.append(f"Giả thuyết từ chiến lược gốc: {parent_reflection.hypothesis}")
            lines.append(f"Tóm tắt hiệu suất: {parent_reflection.summary}")

    next_iter = len(history.strategies)
    lines.append(f"\n=== Nhiệm vụ ===")
    lines.append(
        f"Hãy đề xuất một chiến lược mới bằng cách thay đổi/tối ưu trực tiếp từ chiến lược gốc (Parent Strategy: {parent_strategy.id[:8] if parent_strategy else 'None'}). "
        f"Không tối ưu dựa trên các chiến lược khác hoặc chiến lược gần đây nhất nếu nó khác chiến lược gốc này. "
        f"Đề xuất cho iteration {next_iter}."
    )
    return "\n".join(lines)


def generate_few_shot_reasoning(
    passage: str,
    question: str,
    program: str,
    category: str,
    model,  # QwenInference
    max_attempts: int = 3,
) -> str:
    """
    Generate a full CoT-style response for a programmatic few-shot example.
    """
    from src.executor import normalize_program as _norm

    gold_norm = _norm(program)

    system_message = (
        "Bạn là trợ lý AI chuyên phân tích tài chính tiếng Việt. "
        "Nhiệm vụ của bạn là tạo ra một câu trả lời mẫu (few-shot demonstration) "
        "cho bài toán tài chính, theo đúng định dạng đầu ra mà mô hình phải tạo ra.\n\n"
        "Định dạng bắt buộc:\n"
        "1. Một khối <think>...</think> ngắn gọn (tối đa 100 từ), tập trung vào "
        "công thức toán học và các giá trị cần trích xuất. KHÔNG viết dài dòng.\n"
        "2. Ngay sau </think>, một khối JSON với đúng 3 khóa:\n"
        "{\n"
        "  \"Reasoning\": \"Giải thích 2 câu tiếng Việt: câu 1 nêu giá trị trích xuất, câu 2 giải thích phép tính\",\n"
        "  \"Program syntax\": \"<phải khớp CHÍNH XÁC với chương trình đã cho>\",\n"
        "  \"Numerical result\": <kết quả số cuối cùng>\n"
        "}\n\n"
        "QUAN TRỌNG: Trường 'Program syntax' phải chứa ĐÚNG chương trình đã được cung cấp, không thay đổi."
    )

    for attempt in range(max_attempts):
        user_message = (
            f"Ngữ cảnh:\n{passage}\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Chương trình đúng: {program}\n\n"
            f"Hãy tạo câu trả lời mẫu hoàn chỉnh theo định dạng <think>...</think>{{JSON}} "
            f"với 'Program syntax' phải là CHÍNH XÁC: {program}"
        )
        prompt = model.format_prompt(
            system_message=system_message,
            user_message=user_message,
            enable_thinking=True,
        )
        try:
            raw_output = model.generate_text(prompt, max_new_tokens=512, temperature=0.0)
            extracted = extract_answer(raw_output)
            if extracted and _norm(extracted) == gold_norm:
                logger.debug(
                    "Few-shot CoT verified on attempt %d (gold=%s extracted=%s)",
                    attempt + 1, program, extracted,
                )
                return raw_output.strip()
            else:
                logger.warning(
                    "Few-shot CoT attempt %d/%d: program mismatch "
                    "(gold_norm=%r, extracted_norm=%r) — retrying",
                    attempt + 1, max_attempts,
                    gold_norm, _norm(extracted) if extracted else None,
                )
        except Exception as e:
            logger.warning("Few-shot CoT generation attempt %d failed: %s", attempt + 1, e)

    logger.warning(
        "All %d attempts failed for program %r — using static fallback reasoning",
        max_attempts, program,
    )
    return f"Bài toán thuộc nhóm {category}. Thực hiện phép tính theo chương trình DSL."


def propose_self(
    history: StrategyHistory,
    model,  # QwenInference
    max_retries: int = 5,
    parent_strategy_id: Optional[str] = None,
    train_dataset: Optional[Dataset] = None,
) -> tuple[Strategy, int]:
    """
    Use the Qwen inference model itself to propose a new strategy.

    TODO: Implement strategy proposal and dynamic few-shot selection.
    Steps:
      1. Build propose message.
      2. Call model to generate a free-form proposal.
      3. Clean thinking tags.
      4. Coerce into a JSON ProposerSchema dictionary.
      5. Identify weakest category from reflection.
      6. Select up to 2 matching training examples and generate CoT reasoning for them.
      7. Validate generated/extracted few-shot programs using _is_valid_dsl_program().
      8. Return a new Strategy object and meta token usage.
    """
    propose_message = _build_propose_message(history, parent_strategy_id)
    propose_prompt = model.format_prompt(system_message=_SYSTEM_PROPOSE, user_message=propose_message, enable_thinking=True)
    first_proposal = model.generate_text(propose_prompt)
    clean_first_proposal = re.sub(r"<think>.*?</think>", "", first_proposal, flags=re.DOTALL).strip()
    
    first_cleaned_prompt = model.format_prompt(system_message=_SYSTEM_PROPOSE, user_message=clean_first_proposal, enable_thinking=True)
    json_proposal = model.generate_text(first_cleaned_prompt, guided_json=ProposerSchema.model_json_schema())
    
    try:
        json_output = json.loads(json_proposal)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        json_output = {}
    
    # Identify the weakest category from the latest reflection
    weakest_category = "other"
    target_reflection = None
    target_strategy = None

    if parent_strategy_id is not None:
        for s, r in zip(history.strategies, history.reflections):
            if s.id == parent_strategy_id:
                target_reflection = r
                target_strategy = s
                break
    else:
        target_reflection = history.latest_reflection()
        target_strategy = history.latest_strategy()

    if target_reflection and target_reflection.accuracy_by_type:
        weakest_category = min(target_reflection.accuracy_by_type, key=target_reflection.accuracy_by_type.get)

    # Select up to 2 matching training examples for the weakest category
    few_shot_examples = []
    if train_dataset is not None and weakest_category != "other":
        candidates = [ex for ex in train_dataset if classify_question_type(ex["answer"]) == weakest_category]
        # Prioritize multi-step chains (#1, #2 references) for all arithmetic types — those are the hardest
        if weakest_category in {"addition", "subtraction", "multiplication", "division"}:
            candidates.sort(key=lambda ex: -ex["answer"].count(","))
        matching_examples = [
            FewShotExample(
                passage=ex["context"],
                question=ex["question"],
                answer=ex["answer"],
                reasoning=generate_few_shot_reasoning(
                    ex["context"], ex["question"], ex["answer"], weakest_category, model, max_attempts=max_retries
                )
            )
            for ex in candidates
        ]
    few_shot_examples = matching_examples[:2]
    
    # Validate generated/extracted few-shot programs using _is_valid_dsl_program() and it needs to merge with its parent example if any
    valid_few_shot_examples = target_strategy.few_shot_examples if target_strategy else []
    for example in few_shot_examples:
        if _is_valid_dsl_program(example.answer):
            valid_few_shot_examples.append(example)
        else:
            logger.warning(f"Invalid DSL program in few-shot example: {example.answer}")
    
    # Return a new Strategy object and meta token usage.
    fallback_template = (
        target_strategy.prompt_template if target_strategy
        else "Bạn là chuyên gia phân tích tài chính. Hãy viết chương trình DSL để trả lời câu hỏi."
    )
    prompt_template = json_output.get("instruction_phrasing") or fallback_template

    new_strategy = Strategy(
        id=str(uuid.uuid4()),
        prompt_template=prompt_template,
        cot_format=CoTFormat(json_output.get("cot_format") if json_output.get("cot_format") in _VALID_COT else "none"),
        few_shot_examples=valid_few_shot_examples,
        retrieval_config=RetrievalConfig(
            enabled=True,
            top_k=max_retries,
            similarity_threshold=0.75,
        ),
        metadata=StrategyMetadata(
            iteration=len(history.strategies),
            parent_id=parent_strategy_id,
        )
    )
    total_meta_tokens = model.count_tokens(first_proposal) + model.count_tokens(json_proposal) if bool(json_output) else 0
    return (new_strategy, total_meta_tokens)
