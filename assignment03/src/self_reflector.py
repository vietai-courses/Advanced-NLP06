"""
self_reflector.py — Self-optimization reflector: uses Qwen itself as the meta-agent.

Qwen analyses its own failure cases and generates a hypothesis for improvement.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.executor import EvalResult
from src.strategy import Reflection, Strategy, CoTFormat
from pydantic import BaseModel, Field

class ReflectionSchema(BaseModel):
    accuracy_by_type: dict[str, float] = Field(..., description="Mapping from question category to accuracy (0.0 to 1.0)")
    failure_patterns: list[str] = Field(..., description="List of observed error patterns")
    hypothesis: str = Field(..., description="A concrete, testable hypothesis for why the strategy failed")
    summary: str = Field(..., description="One-paragraph prose summary of the reflection")

logger = logging.getLogger(__name__)

_SYSTEM_REFLECT = """\
Bạn là trợ lý nghiên cứu NLP đang phân tích hiệu suất của một chiến lược prompting \
trên bài toán giải toán tài chính tiếng Việt bằng cách lập chương trình (hàm toán học).

Dựa trên kết quả đánh giá và các lỗi sai, hãy:
1. Nhóm các lỗi sai thành các mẫu lỗi (failure patterns) thay vì chỉ nhìn vào loại phép toán. Các mẫu phổ biến:
   - Lỗi đơn vị/tỷ lệ (vd: kết quả cần tỷ lệ nhưng lại nhân 100 thành phần trăm)
   - Sai logic công thức (vd: câu hỏi hỏi tỷ trọng nhưng dùng phép trừ rồi chia, hoặc ngược lại)
   - Vi phạm cú pháp lồng hàm (vd: divide(table_average(...), x))
   - Trích xuất sai số (vd: lấy số lớn nhất/nổi bật nhất thay vì số đúng ngữ cảnh)
2. Phân tích nguyên nhân sâu xa vì sao prompt hiện tại gây ra các lỗi trên.
3. Đề xuất giả thuyết cải thiện có thể kiểm chứng được cho template prompt.

YÊU CẦU ĐỘ DÀI VÀ CẤU TRÚC (BẮT BUỘC):
- Viết ngắn gọn và súc tích. Tổng độ dài toàn bộ bài phân tích/suy nghĩ KHÔNG ĐƯỢC vượt quá 500 từ.
- KHÔNG lặp từ, không viết dông dài, không tự tạo ra văn bản rác hoặc ký tự lặp vô nghĩa.
- Giả thuyết: tối đa 2 câu.
- Tóm tắt (summary): tối đa 1 đoạn văn ngắn (3-4 câu)."""


def _build_reflect_message(strategy: Strategy, eval_result: EvalResult, progressive: bool = True) -> str:
    lines = [
        f"=== Chiến lược ===",
        f"CoT: {strategy.cot_format.value}",
        f"Template: {strategy.prompt_template[:300]!r}",
        f"\n=== Kết quả đánh giá ===",
        f"Độ chính xác tổng: {eval_result.accuracy:.3f} ({eval_result.num_correct}/{eval_result.num_examples})",
        "\nĐộ chính xác theo loại câu hỏi:",
    ]
    for q_type, acc in sorted(eval_result.accuracy_by_type.items(), key=lambda x: x[1]):
        count = eval_result.count_by_type.get(q_type, 0)
        lines.append(f"  {q_type}: {acc:.3f} ({count} ví dụ)")

    # Check for token budget collapse/truncation
    avg_out_tokens = 0
    if eval_result.per_question:
        avg_out_tokens = sum(r.output_tokens for r in eval_result.per_question) / len(eval_result.per_question)
    
    limit = 1024 if strategy.cot_format != CoTFormat.NONE else 256
    if avg_out_tokens >= 0.9 * limit:
        lines.append(
            "\n[CẢNH BÁO CỰC KỲ QUAN TRỌNG: SỰ CỐ GIỚI HẠN TOKEN / LẶP VÔ HẠN]\n"
            f"Số lượng token đầu ra trung bình mỗi câu hỏi ({avg_out_tokens:.1f}) đã đạt sát giới hạn tối đa ({limit}).\n"
            "Mô hình đã viết quá nhiều giải thích/suy nghĩ dông dài hoặc bị lặp vô hạn và BỊ CẮT GIỮA CHỪNG trước khi kịp trả về chương trình!\n"
            "Để khắc phục, bạn PHẢI tắt Chain-of-Thought (chuyển cot_format thành 'none') HOẶC giới hạn nghiêm ngặt độ dài suy nghĩ (tối đa 2 câu) "
            "và yêu cầu trả về trực tiếp định dạng PROGRAM:.\n"
        )
    if progressive:
        iteration = strategy.metadata.iteration
        if iteration <= 1:
            top_k = 5
        elif iteration == 2:
            top_k = 3
        else:
            top_k = 1
    else:
        top_k = 5

    iteration = strategy.metadata.iteration
    logger.info("Self-Reflector Progressive Context: iteration=%d, selected top_k=%d failures.", iteration, top_k)
    failures = eval_result.failures(top_k=top_k)
    lines.append("\n=== Các lỗi sai tiêu biểu ===")
    for i, f in enumerate(failures, 1):
        if progressive and iteration >= 3:
            passage_text = f.passage[:200] + "..." if len(f.passage) > 200 else f.passage
            output_text = f.raw_output[:100] + "..." if len(f.raw_output) > 100 else f.raw_output
            lines.append(
                f"\nLỗi {i}:\n"
                f"  Ngữ cảnh (Đoạn văn rút gọn): {passage_text}\n"
                f"  Câu hỏi: {f.question}\n"
                f"  Đúng: {f.gold_answer} (Giá trị: {f.gold_val}) | Dự đoán: {f.predicted_answer} (Giá trị: {f.predicted_val})\n"
                f"  Output: {output_text}"
            )
        else:
            lines.append(
                f"\nLỗi {i}:\n"
                f"  Ngữ cảnh (Đoạn văn/Bảng): {f.passage[:1000]}...\n"
                f"  Câu hỏi: {f.question}\n"
                f"  Đúng: {f.gold_answer} (Giá trị: {f.gold_val}) | Dự đoán: {f.predicted_answer} (Giá trị: {f.predicted_val})\n"
                f"  Output: {f.raw_output[:150]}"
            )

    return "\n".join(lines)


def reflect_self(
    strategy: Strategy,
    eval_result: EvalResult,
    model,  # QwenInference
    max_retries: int = 5,
    progressive: bool = True,
) -> tuple[Reflection, int]:
    """
    Use the Qwen inference model itself to reflect on evaluation results.

    TODO: Implement the reflection loop with Pydantic schema validation.
    Steps:
      1. Build reflect message.
      2. Call model.generate_text to perform a first-pass free-form analysis.
      3. Clean thinking tags from the output.
      4. Perform a second-pass coercion request using model.generate_text(..., guided_json=ReflectionSchema.model_json_schema()).
      5. Validate and parse the returned JSON into the Reflection class.
      6. Return Reflection object and estimated token usage.
    """
    # --- YOUR CODE HERE ---
    reflect_message = _build_reflect_message(strategy, eval_result, progressive=progressive)
    reflect_prompt = model.format_prompt(system_message=_SYSTEM_REFLECT, user_message=reflect_message, enable_thinking=True)
    first_pass_output = model.generate_text(reflect_prompt)
    # Strip any thinking tags or extra text from the first pass output
    first_cleaned_output = re.sub(r"<think>.*?</think>", "", first_pass_output, flags=re.DOTALL).strip()
    first_cleaned_prompt = model.format_prompt(system_message=_SYSTEM_REFLECT, user_message=first_cleaned_output, enable_thinking=True)
    second_pass_output = model.generate_text(first_cleaned_prompt, top_k = max_retries, guided_json=ReflectionSchema.model_json_schema())
    try:
        json_output = json.loads(second_pass_output)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        json_output = {}
    
    top_failures = []
    for i, f in enumerate(json_output.get("top_failures", [])):
        top_failures.append({i: f})

    reflection_output = Reflection(
        strategy_id=strategy.id,
        accuracy_by_type=json_output.get("accuracy_by_type", {}),
        top_failures=top_failures,
        hypothesis=json_output.get("hypothesis", "Chiến lược hiện tại yếu nhất"),
        summary=json_output.get("summary", "Fallback summary: Unable to parse reflection output."),
        raw_response=second_pass_output,
    )
    return (reflection_output, model.count_tokens(first_pass_output) + model.count_tokens(second_pass_output) if bool(json_output) else 0)