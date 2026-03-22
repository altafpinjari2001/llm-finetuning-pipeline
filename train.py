"""
LLM Fine-tuning Pipeline - Training Script.

Main entry point for fine-tuning LLMs with LoRA/QLoRA.
"""

import argparse
import logging
import os
from pathlib import Path

import yaml
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from src.model.loader import load_model_and_tokenizer
from src.model.lora_config import apply_lora
from src.data.dataset import DatasetManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    """Execute the fine-tuning pipeline."""
    logger.info("=" * 60)
    logger.info("🧬 LLM Fine-tuning Pipeline")
    logger.info("=" * 60)

    # ── Load Model ───────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        quantization=config["model"].get("quantization", "4bit"),
        token=os.getenv("HF_TOKEN"),
    )

    # ── Apply LoRA ───────────────────────────────────────────
    model = apply_lora(
        model=model,
        model_name=config["model"]["name"],
        rank=config["lora"]["rank"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"].get("dropout", 0.05),
    )

    # ── Prepare Dataset ──────────────────────────────────────
    dataset_manager = DatasetManager(
        tokenizer=tokenizer,
        max_length=config["data"].get("max_length", 2048),
        template_name=config["data"].get("template", "alpaca"),
    )

    train_dataset, val_dataset = dataset_manager.prepare(
        file_path=config["data"]["train_file"],
        val_split=config["data"].get("val_split", 0.1),
    )

    # ── Data Collator ────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ── Training Arguments ───────────────────────────────────
    output_dir = config.get("output_dir", "outputs/model")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"].get(
            "gradient_accumulation_steps", 4
        ),
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"].get(
            "scheduler", "cosine"
        ),
        warmup_ratio=config["training"].get("warmup_ratio", 0.1),
        weight_decay=config["training"].get("weight_decay", 0.01),
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=config.get("report_to", "wandb"),
        run_name=config.get("run_name", "llm-finetune"),
    )

    # ── Train ────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✅ Model saved to: {output_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Fine-tuning Pipeline"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
