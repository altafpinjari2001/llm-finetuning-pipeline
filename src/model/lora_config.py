"""
LLM Fine-tuning Pipeline - LoRA Configuration.

Configures Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
"""

import logging

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# Default target modules for common architectures
TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense"],
    "default": ["q_proj", "v_proj"],
}


def get_target_modules(model_name: str) -> list[str]:
    """Determine target modules based on model architecture."""
    model_lower = model_name.lower()
    for key, modules in TARGET_MODULES.items():
        if key in model_lower:
            return modules
    return TARGET_MODULES["default"]


def apply_lora(
    model: PreTrainedModel,
    model_name: str,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
    bias: str = "none",
) -> PreTrainedModel:
    """
    Apply LoRA adapter to the model.

    Args:
        model: Base pre-trained model.
        model_name: Model name for auto-detecting target modules.
        rank: LoRA rank (higher = more capacity, more params).
        alpha: LoRA alpha scaling factor.
        dropout: Dropout probability for LoRA layers.
        target_modules: Specific module names to apply LoRA.
        bias: Bias training strategy.

    Returns:
        Model with LoRA adapters applied.
    """
    # Prepare for quantized training
    model = prepare_model_for_kbit_training(model)

    # Auto-detect target modules if not specified
    if target_modules is None:
        target_modules = get_target_modules(model_name)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        f"LoRA applied: {trainable:,} trainable params "
        f"({100 * trainable / total:.2f}% of {total:,} total)"
    )

    return model
