"""
LLM Fine-tuning Pipeline - Model Loader.

Handles loading base models and tokenizers with quantization.
"""

import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    quantization: Optional[str] = "4bit",
    use_flash_attention: bool = True,
    trust_remote_code: bool = True,
    token: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a pre-trained model and tokenizer with optional quantization.

    Args:
        model_name: HuggingFace model ID or local path.
        quantization: "4bit", "8bit", or None.
        use_flash_attention: Whether to use Flash Attention 2.
        trust_remote_code: Trust remote code for custom models.
        token: HuggingFace token for gated models.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading model: {model_name}")

    # Quantization config
    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit QLoRA quantization")
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization")

    # Load model
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
        "token": token,
    }

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )
    except Exception:
        # Fallback without flash attention
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )
        logger.warning("Flash Attention not available, using default")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        token=token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    logger.info(
        f"Model loaded: {model.num_parameters() / 1e6:.1f}M params"
    )
    return model, tokenizer
