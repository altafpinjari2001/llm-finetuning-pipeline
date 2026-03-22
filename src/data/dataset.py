"""
LLM Fine-tuning Pipeline - Dataset Module.

Handles dataset loading, formatting, and tokenization
for instruction fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from .templates import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset preparation for LLM fine-tuning."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        template_name: str = "alpaca",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = PROMPT_TEMPLATES.get(
            template_name, PROMPT_TEMPLATES["alpaca"]
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(
            f"DatasetManager initialized "
            f"(max_length={max_length}, template={template_name})"
        )

    def load_from_jsonl(self, file_path: str) -> Dataset:
        """
        Load dataset from a JSONL file.

        Expected format per line:
        {"instruction": "...", "input": "...", "output": "..."}
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))

        dataset = Dataset.from_list(data)
        logger.info(f"Loaded {len(dataset)} examples from {path.name}")
        return dataset

    def load_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
    ) -> Dataset:
        """Load dataset from HuggingFace Hub."""
        dataset = load_dataset(
            dataset_name, subset, split=split
        )
        logger.info(
            f"Loaded {len(dataset)} examples from {dataset_name}"
        )
        return dataset

    def format_prompt(self, example: dict) -> str:
        """Format a single example using the prompt template."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            prompt = self.template["with_input"].format(
                instruction=instruction,
                input=input_text,
                output=output,
            )
        else:
            prompt = self.template["without_input"].format(
                instruction=instruction,
                output=output,
            )
        return prompt

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize and format the entire dataset."""

        def _tokenize(example):
            prompt = self.format_prompt(example)
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(
            _tokenize,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        logger.info(
            f"Tokenized {len(tokenized_dataset)} examples "
            f"(max_length={self.max_length})"
        )
        return tokenized_dataset

    def prepare(
        self,
        file_path: str,
        val_split: float = 0.1,
    ) -> tuple[Dataset, Dataset]:
        """
        End-to-end dataset preparation.

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        dataset = self.load_from_jsonl(file_path)
        tokenized = self.tokenize_dataset(dataset)

        split = tokenized.train_test_split(test_size=val_split)
        logger.info(
            f"Split: {len(split['train'])} train, "
            f"{len(split['test'])} validation"
        )
        return split["train"], split["test"]
