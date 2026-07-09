"""
model.py — QwenInference: SGLang-based inference wrapper for Qwen3.5-4B-AWQ.

Uses SGLang for high-throughput batched inference. SGLang is 5-10x faster than
the HuggingFace generate() pipeline on the same hardware.

For A10G (24 GB VRAM), use an AWQ-quantized model:
    QuantTrio/Qwen3.5-4B-AWQ  (~3 GB weights, leaves room for KV cache)

Answer extraction handles extraction of mathematical programs:
    e.g., "subtract(108.50, 100), divide(#0, 100)"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """Holds the raw text output and token counts for a single example."""

    raw_output: str
    predicted_answer: Optional[str]  # "A", "B", "C", "D", or None
    input_tokens: int
    output_tokens: int


class QwenInference:
    """
    SGLang-backed inference wrapper for Qwen3.5-Instruct models.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID or local path. For A10G GPUs use the AWQ variant:
        "QuantTrio/Qwen3.5-4B-AWQ"

    max_new_tokens:
        Maximum tokens generated per example.
    temperature:
        Sampling temperature. 0.0 = greedy decoding.
    use_4bit:
        If True and the model name doesn't already indicate quantization,
        logs a warning. For A10G, pass an AWQ/GPTQ model name instead.
    gpu_memory_utilization:
        Fraction of GPU memory SGLang may use for weights + KV cache (0–1).
    max_model_len:
        Maximum sequence length (prompt + completion). Longer sequences are
        truncated. 2048 is sufficient for typical QA passages.
    """

    def __init__(
        self,
        model_name_or_path: str = "QuantTrio/Qwen3.5-4B-AWQ",

        max_new_tokens: int = 256,
        temperature: float = 0.0,
        use_4bit: bool = True,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_4bit = use_4bit
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        self._llm = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model via SGLang. Call once before inference."""
        import sglang as sgl
        from transformers import AutoTokenizer

        name = self.model_name_or_path

        logger.info(
            "Loading SGLang engine: model=%s gpu_mem=%.0f%%",
            name,
            self.gpu_memory_utilization * 100,
        )

        self._llm = sgl.Engine(
            model_path=name,
            mem_fraction_static=self.gpu_memory_utilization,
            context_length=self.max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
        )

        # Load tokenizer separately for apply_chat_template / count_tokens.
        self._tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Model loaded successfully via SGLang.")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def generate_batch(self, prompts: list[str], cot_format: bool = False) -> list[GenerationResult]:
        """
        Run generation for a list of already-formatted prompt strings.

        SGLang handles batching and scheduling internally — pass all prompts
        at once for maximum throughput.
        """
        if self._llm is None:
            raise RuntimeError("Call load() before generate_batch().")

        if self.temperature > 0.0:
            if cot_format:
                # Thinking mode for precise coding tasks (e.g. CoT Program Generation)
                temp = 0.6 if self.temperature == 1.0 or self.temperature == 0.7 else self.temperature
                t_p = 0.95
                t_k = 20
                m_p = 0.0
                p_p = 1.5  # presence penalty prevents repetition loops in long thinking blocks
            else:
                # Instruct (or non-thinking) mode for reasoning tasks
                temp = self.temperature
                t_p = 0.95
                t_k = 20
                m_p = 0.0
                p_p = 1.5
        else:
            if cot_format:
                # Greedy + thinking: use a tiny temperature to unlock presence_penalty
                temp = 0.01
                t_p = 1.0
                t_k = -1
                m_p = 0.0
                p_p = 1.5  # critical: prevents infinite loops in thinking mode at near-greedy decoding
            else:
                temp = 0.0
                t_p = 1.0
                t_k = -1
                m_p = 0.0
                p_p = 0.0

        sampling_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": temp,
            "top_p": t_p,
            "top_k": t_k,
            "min_p": m_p,
            "presence_penalty": p_p,
            "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        }

        # Truncate prompts that exceed the context budget.
        max_input = self.max_model_len - self.max_new_tokens
        if max_input <= 0:
            raise ValueError(
                f"max_model_len ({self.max_model_len}) must be strictly greater than "
                f"max_new_tokens ({self.max_new_tokens}) to leave room for the prompt."
            )
        truncated = []
        for p in prompts:
            ids = self._tokenizer.encode(p, add_special_tokens=False)
            if len(ids) > max_input:
                ids = ids[:max_input]
                p = self._tokenizer.decode(ids, skip_special_tokens=True)
            truncated.append(p)

        outputs = self._llm.generate(truncated, sampling_params)

        results: list[GenerationResult] = []
        for out in outputs:
            raw_text = out["text"].strip()
            predicted = extract_answer(raw_text)
            input_tokens = out["meta_info"]["prompt_tokens"]
            output_tokens = out["meta_info"]["completion_tokens"]
            results.append(
                GenerationResult(
                    raw_output=raw_text,
                    predicted_answer=predicted,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )

        return results

    def format_prompt(
        self,
        system_message: str,
        user_message: str,
        enable_thinking: bool = False,
    ) -> str:
        """
        Build a ChatML-formatted string for Qwen-Instruct using the
        tokenizer's apply_chat_template().

        For Qwen3 models, enable_thinking=False (default) injects the /no_think
        control token which disables the verbose reasoning block and produces
        direct answers — critical for eval throughput and token budget.
        Set enable_thinking=True only for meta-agent tasks (reflection/proposing)
        where reasoning quality matters more than brevity.
        """
        if self._tokenizer is None:
            raise RuntimeError("Call load() before format_prompt().")
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older tokenizers (Qwen2.5) don't support enable_thinking — fall back
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 8192,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        guided_json: Optional[dict | str] = None,
    ) -> str:
        """
        Single unconstrained text generation — used by self_proposer / self_reflector.

        Unlike generate_batch() which is optimised for high-throughput MC inference,
        this is for low-volume, long-form generation (strategy proposals, reflections).
        Returns the raw generated text string.
        """
        if self._llm is None:
            raise RuntimeError("Call load() before generate_text().")

        # For guided JSON, enforce deterministic decoding
        if guided_json is not None:
            temp = 0.0
            t_p = 1.0
            t_k = -1
            m_p = 0.0
            p_p = 0.0
        else:
            temp = temperature
            t_p = top_p
            t_k = top_k
            m_p = min_p
            p_p = presence_penalty

        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temp,
            "top_p": t_p,
            "top_k": t_k,
            "min_p": m_p,
            "presence_penalty": p_p,
            "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        }

        if guided_json is not None:
            if isinstance(guided_json, dict):
                import json
                sampling_params["json_schema"] = json.dumps(guided_json)
            else:
                sampling_params["json_schema"] = guided_json

        max_input = self.max_model_len - max_new_tokens
        ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > max_input:
            ids = ids[:max_input]
            prompt = self._tokenizer.decode(ids, skip_special_tokens=True)

        output = self._llm.generate(prompt, sampling_params)
        return output["text"].strip()

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError("Call load() before count_tokens().")
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None


# ------------------------------------------------------------------
# Answer extraction (module-level for testability)
# ------------------------------------------------------------------


def clean_unnecessary_parentheses(program: str) -> str:
    if not program:
        return program
    chars = list(program)
    stack = []
    pairs = []
    for idx, char in enumerate(chars):
        if char == '(':
            stack.append(idx)
        elif char == ')':
            if stack:
                start_idx = stack.pop()
                pairs.append((start_idx, idx))
    to_delete = set()
    for start, end in pairs:
        prev_idx = start - 1
        while prev_idx >= 0 and chars[prev_idx].isspace():
            prev_idx -= 1
        is_func_call = False
        if prev_idx >= 0:
            c = chars[prev_idx]
            if c.isalnum() or c == '_':
                is_func_call = True
        if not is_func_call:
            to_delete.add(start)
            to_delete.add(end)
    cleaned = "".join(chars[i] for i in range(len(chars)) if i not in to_delete)
    cleaned = re.sub(r',\s*,', ', ', cleaned)
    cleaned = cleaned.strip().strip(',')
    return cleaned

def fix_bare_hashes(program: str) -> str:
    if not program:
        return program
    chars = list(program)
    step_count = 0
    i = 0
    while i < len(chars):
        if chars[i] == ')':
            step_count += 1
        elif chars[i] == '#':
            if i + 1 >= len(chars) or not chars[i + 1].isdigit():
                prev_step_idx = max(0, step_count - 1)
                replacement = str(prev_step_idx)
                chars.insert(i + 1, replacement)
                i += len(replacement)
        i += 1
    return "".join(chars)

def clean_and_fix_program(program: str) -> str:
    if not program:
        return program
    program = clean_unnecessary_parentheses(program)
    program = fix_bare_hashes(program)
    return program

def extract_answer(text: str) -> Optional[str]:
    """
    Extract a mathematical program string from a model's free-form output.

    Handles multi-line programs, balanced nested parentheses, prefixes,
    and Qwen3-style thinking blocks (<think>...</think> or "Thinking Process:").
    """
    if not text:
        return None

    # First check if there is a </think> block.
    # If so, the JSON block must be after the </think> tag.
    content_after_think = text
    if "</think>" in text:
        parts = text.split("</think>")
        content_after_think = parts[-1].strip()

    # Try to extract from JSON format first
    json_match = re.search(r"\{.*\}", content_after_think, re.DOTALL)
    if json_match:
        try:
            import json
            data = json.loads(json_match.group(), strict=False)
            program_val = None
            for key in ["Program syntax", "program syntax", "Program", "program", "program_syntax"]:
                if key in data:
                    program_val = data[key]
                    break
            if not program_val:
                for key, val in data.items():
                    if "program" in key.lower() or "syntax" in key.lower():
                        program_val = val
                        break
            if program_val and isinstance(program_val, str):
                program_val = program_val.strip()
                program_val = re.sub(r"^`+", "", program_val)
                program_val = re.sub(r"`+$", "", program_val).strip()
                if program_val:
                    return clean_and_fix_program(program_val)
        except Exception:
            pass

    # Fallback: operate on content_after_think for the rest of extraction
    text = content_after_think

    # ---- 0. Strip Qwen3 thinking blocks ----
    # Remove <think>...</think> blocks (Qwen3 native thinking format)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    # ---- High-priority strict PROGRAM: extraction ----
    program_match = re.search(r"PROGRAM:\s*(.*)", text, re.IGNORECASE)
    if program_match:
        # Extract the line containing the program
        prog_text = program_match.group(1).strip().split("\n")[0].strip()
        # Clean markdown backticks if any
        prog_text = re.sub(r"^`+", "", prog_text)
        prog_text = re.sub(r"`+$", "", prog_text).strip()
        if prog_text:
            return clean_and_fix_program(prog_text)

    # If there's an explicit label like "Chương trình: X" or "**Answer:**\n X",
    # extract only what comes after it. This handles Thinking Process blocks that
    # end with a clearly labeled answer.
    explicit_label = re.search(
        r"(?:ch\u01b0\u01a1ng\s*tr\u00ecnh|program|\*\*answer\*\*|\*\*\u0111\u00e1p\s*\u00e1n\*\*)\s*[:\-]?\s*\n*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if explicit_label:
        text = explicit_label.group(1).strip()
    else:
        # Greedy strip: if "Thinking Process:" is present, drop everything up to
        # a double newline that *immediately* precedes a line starting with a
        # known operation (i.e. the final answer paragraph).
        if re.search(r"Thinking Process:", text, re.IGNORECASE):
            # Try to find the LAST paragraph that contains only program tokens
            # (no bullet points, no markdown headers)
            paragraphs = re.split(r"\n{2,}", text)
            clean_paragraphs = []
            for para in paragraphs:
                para = para.strip()
                # Keep paragraphs that look like programs (function calls) and
                # NOT like thinking step headers (numbered lists, markdown headers)
                if para and not re.match(r"^(\d+\.|#+|\*\*|Thinking Process)", para):
                    clean_paragraphs.append(para)
            text = "\n\n".join(clean_paragraphs) if clean_paragraphs else text

    text = text.strip()

    if not text:
        return None

    # Operations list
    ops = {"add", "subtract", "multiply", "divide", "table_average", "table_max", "table_min", "table_sum", "abs"}

    # 1. Strip markdown code blocks
    text = re.sub(r"```(?:[a-zA-Z_0-9]*)\n(.*?)```", r"\1", text, flags=re.DOTALL)

    # 2. Clean up "Chương trình:", "Program:", "Đáp án:", "Answer:" prefixes
    text = re.sub(
        r"^(?:chương\s*trình|program|đáp\s*án|answer)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # 3. Parse line by line to extract valid program components
    lines = text.split("\n")
    valid_parts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that look like backtick-formatted function lists
        # (system-prompt leakage: "`add`, `subtract`, `multiply`...")
        backtick_ops = re.findall(r"`([a-zA-Z_]+)`", line)
        if len(backtick_ops) >= 2 and all(
            op in ops for op in backtick_ops
        ):
            continue

        # Look for operations or references
        has_op = any(op in line.lower() for op in ops)
        has_ref = any(f"#{i}" in line for i in range(10))

        if has_op or has_ref:
            # Find the starting index of the expression (first operation word or reference/parenthesis)
            first_idx = len(line)
            for op in ops:
                idx = line.lower().find(op)
                if idx != -1 and idx < first_idx:
                    first_idx = idx

            if first_idx == len(line):
                # Search for any operation prefix or opening parenthesis
                op_match = re.search(r"[a-zA-Z_0-9]+(?=\()", line)
                if op_match:
                    first_idx = op_match.start()
                else:
                    first_idx = 0

            # Adjust first_idx backwards to include any leading parentheses/spaces/commas
            while first_idx > 0 and line[first_idx - 1] in {'(', ' ', ','}:
                if line[first_idx - 1].isalnum() or line[first_idx - 1] == '_':
                    break
                first_idx -= 1

            part = line[first_idx:].strip()
            part = clean_and_fix_program(part)
            if part:
                # We scan character by character to extract balanced chunks
                stack = 0
                chunk_start = 0
                has_paren = False
                for i, char in enumerate(part):
                    if char == '(':
                        stack += 1
                        has_paren = True
                    elif char == ')':
                        stack -= 1
                        has_paren = True
                        if stack == 0:
                            # We found a complete balanced chunk!
                            chunk = part[chunk_start:i+1].strip()
                            if chunk:
                                valid_parts.append(chunk)
                            # Now search for the next operation start after this chunk
                            next_idx = i + 1
                            while next_idx < len(part) and part[next_idx] in {',', ' ', '\n', '\t'}:
                                next_idx += 1
                            chunk_start = next_idx

                # Fallback: if we didn't find any balanced paren but operation keywords are present
                if not has_paren:
                    valid_parts.append(part)

    if valid_parts:
        return clean_and_fix_program(", ".join(valid_parts))

    # Fallback: Check if the text itself matches a program-like layout
    if "(" in text and ")" in text:
        return clean_and_fix_program(text.strip())

    return None

