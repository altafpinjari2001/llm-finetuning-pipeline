"""
LLM Fine-tuning Pipeline - Prompt Templates.

Standardized prompt templates for instruction fine-tuning.
"""

PROMPT_TEMPLATES = {
    "alpaca": {
        "with_input": (
            "Below is an instruction that describes a task, "
            "paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        ),
        "without_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:\n{output}"
        ),
    },
    "chatml": {
        "with_input": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n"
            "<|im_start|>assistant\n{output}<|im_end|>"
        ),
        "without_input": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n{output}<|im_end|>"
        ),
    },
    "llama3": {
        "with_input": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful AI assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}\n\n{input}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{output}<|eot_id|>"
        ),
        "without_input": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful AI assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{output}<|eot_id|>"
        ),
    },
}
