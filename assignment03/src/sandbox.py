"""
sandbox.py - Stage 0: Run a single zero-shot prediction and evaluate baseline accuracy.
"""

from src.data import load_data_splits
from src.model import QwenInference

# Globally cached model instance to prevent duplicate engine initialization
_cached_model = None

SANDBOX_SYSTEM_MESSAGE = (
    "Ban la mot tro ly AI chuyen phan tich tai chinh tieng Viet. "
    "Nhiem vu cua ban la viet chuong trinh DSL de tra loi cau hoi dua tren van ban va bang so lieu. "
    "Chi tra ve mot dong duy nhat theo dung dinh dang: PROGRAM: <dsl_program>."
)

SANDBOX_PROMPT_TEMPLATE = (
    "Hay giai bai toan tai chinh sau bang cach viet chuong trinh dang ham toan hoc.\n\n"
    "Boi canh:\n{passage}\n\n"
    "Cau hoi: {question}\n\n"
    "Yeu cau: chi tra ve mot dong duy nhat theo dinh dang PROGRAM: <dsl_program>."
)


def get_model(model_name: str) -> QwenInference:
    """Get or initialize the cached QwenInference instance."""
    global _cached_model
    if _cached_model is None:
        _cached_model = QwenInference(model_name_or_path=model_name, gpu_memory_utilization=0.7)
        _cached_model.load()
    return _cached_model


def is_program_correct(pred_text: str, gold: str, table: list[list[str]], exe_ans: str) -> bool:
    """Verify if a generated program is correct by exact match or executed value match."""
    from src.model import extract_answer
    from src.evaluator import evaluate_program

    extracted = extract_answer(pred_text)
    if not extracted:
        return False

    def normalize(program: str) -> str:
        return "".join(program.split()).lower()

    if normalize(extracted) == normalize(gold):
        return True

    if exe_ans:
        try:
            gold_val = float(exe_ans)
            pred_val = evaluate_program(extracted, table)
            if abs(pred_val - gold_val) <= 1e-4:
                return True
        except Exception:
            pass

    return False


def _build_sandbox_prompt(model: QwenInference, passage: str, question: str) -> str:
    """Build a deterministic prompt that encourages direct DSL output."""
    user_message = SANDBOX_PROMPT_TEMPLATE.format(passage=passage, question=question)
    return model.format_prompt(
        system_message=SANDBOX_SYSTEM_MESSAGE,
        user_message=user_message,
        enable_thinking=False,
    )


def run_sandbox_prediction(model_name: str = "QuantTrio/Qwen3.5-4B-AWQ") -> tuple[str, str]:
    """
    Run a zero-shot prediction on the first training example.

    TODO: Implement the Stage 0 sandbox baseline.

    Goal:
      Run one deterministic zero-shot prediction on the first training example
      and return:
        1. the model's raw text output
        2. the gold DSL program from the dataset

    Recommended steps:
      1. Load a tiny dataset slice with:
           load_data_splits(train_size=1, dev_size=1)
      2. Take the first example from the train split.
      3. Read these fields from the example:
           - context
           - question
           - answer
         The `answer` field is the gold program you should return.
      4. Load the model with:
           get_model(model_name)
      5. Build the sandbox prompt with:
           _build_sandbox_prompt(model, passage, question)
         This helper already applies the prompt template and disables long
         reasoning mode so the output is easier to extract later.
      6. Generate one deterministic response with:
           model.generate_text(prompt, max_new_tokens=256, temperature=0.0)
      7. Return:
           (raw_model_response, gold_program)

    Notes:
      - Return the raw model output exactly as generated.
      - Do not extract the DSL program inside this function.
      - Keep the generation deterministic with temperature=0.0.
    """
    # --- YOUR CODE HERE ---
    train_split, _ = load_data_splits(train_size=1, dev_size=1)
    first_example = train_split[0]
    passage = first_example.get("context") or ""
    question = first_example.get("question") or ""
    gold_program = first_example.get("answer") or ""

    model = get_model(model_name)
    prompt = _build_sandbox_prompt(model, passage, question)
    raw_model_response = model.generate_text(prompt, max_new_tokens=256, temperature=0.0)

    return raw_model_response, gold_program


def run_sandbox_accuracy_check(model: QwenInference, dev_size: int = 50) -> dict:
    """Evaluate zero-shot baseline accuracy on the development split."""
    _, dev_split = load_data_splits(train_size=1, dev_size=dev_size)
    print(f"\nEvaluating baseline accuracy on {len(dev_split)} dev examples...")

    prompts = []
    for ex in dev_split:
        passage = ex.get("context") or ""
        question = ex.get("question") or ""
        prompts.append(_build_sandbox_prompt(model, passage, question))

    results = model.generate_batch(prompts, cot_format=False)

    num_correct = 0
    samples = []
    for idx, (ex, res) in enumerate(zip(dev_split, results)):
        gold_program = ex.get("answer") or ""
        table = ex.get("table") or []
        exe_ans = ex.get("exe_ans") or ""
        predicted_answer = res.predicted_answer or res.raw_output
        correct = is_program_correct(predicted_answer, gold_program, table, exe_ans)
        if correct:
            num_correct += 1

        samples.append(
            {
                "index": idx,
                "passage": ex.get("context") or "",
                "question": ex.get("question") or "",
                "gold_program": gold_program,
                "predicted_answer": predicted_answer,
                "pred_raw": res.raw_output,
                "is_correct": correct,
            }
        )

    accuracy = num_correct / len(dev_split) if dev_split else 0.0
    print("=== Zero-Shot Baseline Results ===")
    print(f"Accuracy: {accuracy * 100:.2f}% ({num_correct}/{len(dev_split)})")

    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_examples": len(dev_split),
        "samples": samples,
    }


if __name__ == "__main__":
    pred, gold = run_sandbox_prediction()
    print("\n=== Model Output ===")
    print(pred)
    print("\n=== Gold Program ===")
    print(gold)
