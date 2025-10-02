#!/usr/bin/env python
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generic training script for popular vision-language models.

This script provides a single entry point for fine-tuning a collection of
vision-language models (VLMs) that are available on the Hugging Face Hub.  The
current implementation targets the following checkpoints out of the box:

* ``Salesforce/blip2-itm-vit-g-coco``
* ``Salesforce/blip2-flan-t5-xl-coco``
* ``llava-hf/llava-1.5-7b-hf``
* ``llava-hf/llava-v1.6-mistral-7b-hf``
* ``Salesforce/instructblip-flan-t5-xl``
* ``openflamingo/OpenFlamingo-3B-vitl-mpt1b``

The script is intentionally designed to be extensible – models that inherit from
``LlavaConfig``, ``Blip2Config`` or ``OpenFlamingoConfig`` can be supported by
simply pointing the ``--model_id`` argument to the desired checkpoint.

Dataset format
--------------

The script expects the training and validation data to be stored in any format
that the 🤗 `datasets.load_dataset` helper can read (``.json``, ``.jsonl``,
``.csv``, ``.parquet``…). Each row must contain at least three columns:

``image``
    Absolute or relative path/URL that points to the image file on disk.  If the
    paths are relative you can provide ``--image_root`` so they are resolved
    against a shared directory.
``instruction``
    Natural-language prompt or question that should condition the model.
``output``
    Expected textual response that the model should produce.  For causal
    decoders this text is appended to the prompt (separated by the
    ``--response_template`` value) before computing the loss.

You can freely rename the columns by passing the ``--image_column``,
``--prompt_column``, and ``--response_column`` arguments.  The example below
demonstrates a JSONL file with the default column names:

.. code-block:: json

    {"image": "images/0001.jpg", "instruction": "Describe the image.", "output": "A cat is sitting on a sofa."}
    {"image": "images/0002.jpg", "instruction": "What is the person doing?", "output": "They are riding a bicycle."}

To launch training with such a dataset you can run:

.. code-block:: bash

    python scripts/train_vlm.py \
        --model_id "llava-hf/llava-1.5-7b-hf" \
        --train_file /path/to/train.jsonl \
        --validation_file /path/to/valid.jsonl \
        --image_column image \
        --prompt_column instruction \
        --response_column output \
        --output_dir ./outputs/llava \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --num_train_epochs 3
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, MutableMapping, Optional, Sequence

import torch
from datasets import DatasetDict, load_dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.trainer_utils import set_seed
from transformers.utils import logging

try:  # pragma: no cover - optional imports that depend on extras
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:  # pragma: no cover - graceful fallback if PEFT is unavailable
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    prepare_model_for_kbit_training = None  # type: ignore

try:  # pragma: no cover - optional imports that are only needed for specific models
    from transformers import AutoModelForVision2Seq, OpenFlamingoForConditionalGeneration
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("A modern version of `transformers` is required.") from exc

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    """Arguments that control how the base model is loaded."""

    model_id: str = field(
        metadata={
            "help": "Model identifier from the Hugging Face Hub."
        }
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Optional model revision to use (branch, tag or commit)."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional cache directory for model weights."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Enable Low-Rank Adaptation (LoRA) fine-tuning."},
    )
    lora_rank: int = field(
        default=16,
        metadata={"help": "Rank used by the LoRA adapters."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Alpha parameter of the LoRA adapters."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout applied inside the LoRA adapters."},
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma separated list of module names that should be wrapped with"
                " LoRA adapters. When omitted a sensible default is chosen based on"
                " the selected model."
            )
        },
    )


@dataclass
class DataArguments:
    """Arguments that describe the vision-language dataset."""

    train_file: str = field(
        metadata={"help": "Path to the training dataset (JSON/JSONL/CSV/Parquet)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to a validation dataset."},
    )
    image_column: str = field(
        default="image",
        metadata={"help": "Column name that stores image paths or URLs."},
    )
    prompt_column: str = field(
        default="instruction",
        metadata={"help": "Column name that stores the instruction / input text."},
    )
    response_column: str = field(
        default="output",
        metadata={"help": "Column name that stores the expected model response."},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional base directory that is joined with relative entries in"
                " the image column."
            )
        },
    )
    max_source_length: int = field(
        default=512,
        metadata={"help": "Maximum tokenised length for the textual prompt."},
    )
    max_target_length: int = field(
        default=512,
        metadata={"help": "Maximum tokenised length for the expected response."},
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of preprocessing workers used by datasets.map."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the number of training examples (useful for debugging)."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the number of evaluation examples (useful for debugging)."},
    )
    response_template: str = field(
        default="\n### Response:\n",
        metadata={
            "help": (
                "Template that separates the prompt from the target text when"
                " training causal decoder only models."
            )
        },
    )


def _resolve_image_path(path: str, root: Optional[str]) -> str:
    if root is None or os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _load_image(path: str) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _infer_data_files(train_file: str, validation_file: Optional[str]) -> Dict[str, str]:
    data_files: Dict[str, str] = {"train": train_file}
    if validation_file is not None:
        data_files["validation"] = validation_file
    return data_files


def _infer_dataset_loader(train_file: str) -> str:
    extension = os.path.splitext(train_file)[1].lower()
    if extension in {".json", ".jsonl"}:
        return "json"
    if extension == ".csv":
        return "csv"
    if extension in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(
        "Unsupported dataset format. Expected one of JSON/JSONL/CSV/Parquet, got"
        f" {extension}."
    )


def _default_lora_targets(model_type: str) -> Sequence[str]:
    # LoRA targets work well across the supported checkpoints – they only affect
    # the projection layers of the text decoder.
    if model_type in {"blip-2", "instructblip"}:
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    if model_type in {"llava", "open-flamingo"}:
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    # Fallback – users can always provide their own overrides via CLI arguments.
    return ("q_proj", "k_proj", "v_proj", "o_proj")


def _select_model_class(model_type: str) -> type[PreTrainedModel]:
    if model_type in {"blip-2", "instructblip", "llava"}:
        return AutoModelForVision2Seq
    if model_type == "open-flamingo":
        return OpenFlamingoForConditionalGeneration
    raise ValueError(f"Unsupported model type: {model_type}")


def _select_processor_class(model_type: str) -> type[ProcessorMixin]:
    if model_type == "open-flamingo":
        from transformers import OpenFlamingoProcessor

        return OpenFlamingoProcessor
    return AutoProcessor


def _maybe_prepare_lora(
    model: PreTrainedModel,
    model_args: ModelArguments,
    model_type: str,
) -> PreTrainedModel:
    if not model_args.use_lora:
        return model
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError(
            "LoRA was requested but PEFT is not installed. Please install the"
            " `peft` package to enable adapter training."
        )

    target_modules: Sequence[str]
    if model_args.lora_target_modules:
        target_modules = tuple(
            module.strip() for module in model_args.lora_target_modules.split(",") if module.strip()
        )
    else:
        target_modules = _default_lora_targets(model_type)

    if any(getattr(model, attr, False) for attr in {"is_loaded_in_4bit", "is_loaded_in_8bit"}):
        if prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
    )
    logger.info("Enabling LoRA adapters on modules: %s", ", ".join(target_modules))
    return get_peft_model(model, lora_config)


class VisionLanguageDataCollator:
    """Pads a batch of features produced by the VLM preprocessors."""

    def __init__(
        self,
        processor: ProcessorMixin,
        is_encoder_decoder: bool,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.processor = processor
        self.is_encoder_decoder = is_encoder_decoder
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer = getattr(processor, "tokenizer", None) or getattr(
            processor, "text_tokenizer", None
        )
        if self.tokenizer is None:
            raise ValueError("Processor does not expose a tokenizer instance.")

    def _pad_tensor_list(
        self,
        tensors: Sequence[torch.Tensor],
        padding_value: int,
    ) -> torch.Tensor:
        batch = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        if self.pad_to_multiple_of is None:
            return batch
        target_length = math.ceil(batch.size(1) / self.pad_to_multiple_of) * self.pad_to_multiple_of
        if target_length == batch.size(1):
            return batch
        pad_size = target_length - batch.size(1)
        pad_tensor = torch.full(
            (batch.size(0), pad_size),
            padding_value,
            dtype=batch.dtype,
            device=batch.device,
        )
        return torch.cat([batch, pad_tensor], dim=1)

    def __call__(self, features: List[MutableMapping[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}
        pixel_key = "pixel_values"
        if pixel_key not in features[0] and "vision_x" in features[0]:
            pixel_key = "vision_x"

        if pixel_key in features[0]:
            pixel_values = [feature[pixel_key] for feature in features]
            if isinstance(pixel_values[0], list):
                # OpenFlamingo returns a list of frame tensors.
                stacked = [torch.stack(item) for item in pixel_values]
                batch[pixel_key] = self._pad_tensor_list(stacked, 0)
            else:
                batch[pixel_key] = torch.stack(pixel_values)

        text_features = []
        for feature in features:
            item = {
                key: value
                for key, value in feature.items()
                if key in {"input_ids", "attention_mask"}
            }
            text_features.append(item)

        padded = self.tokenizer.pad(
            text_features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch.update(padded)

        if "labels" in features[0]:
            labels = [feature["labels"] for feature in features]
            batch["labels"] = self._pad_tensor_list(labels, -100)

        # Collect any additional keys (e.g. Q-Former inputs for BLIP-2).
        for key in features[0].keys():
            if key in {pixel_key, "input_ids", "attention_mask", "labels"}:
                continue
            values = [feature[key] for feature in features]
            if isinstance(values[0], torch.Tensor):
                if values[0].dim() == 0:
                    batch[key] = torch.stack(values)
                elif values[0].dim() == 1:
                    batch[key] = self._pad_tensor_list(values, 0)
                else:
                    batch[key] = torch.stack(values)
            else:
                batch[key] = torch.tensor(values)
        return batch


def _format_prompt(prompt: str, template: str) -> str:
    prompt = prompt.rstrip()
    return f"{prompt}{template}"


def _preprocess_function(
    example: MutableMapping[str, Any],
    *,
    processor: ProcessorMixin,
    model_type: str,
    is_encoder_decoder: bool,
    image_column: str,
    prompt_column: str,
    response_column: str,
    image_root: Optional[str],
    response_template: str,
    max_source_length: int,
    max_target_length: int,
) -> Dict[str, Any]:
    image_path = _resolve_image_path(str(example[image_column]), image_root)
    image = _load_image(image_path)

    prompt_text = str(example[prompt_column])
    response_text = str(example[response_column])

    if model_type == "open-flamingo":
        processor_inputs = processor(
            text=[prompt_text],
            images=[[image]],
            return_tensors="pt",
            truncation=True,
            max_length=max_source_length + max_target_length,
        )
    else:
        processor_inputs = processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=max_source_length,
        )

    features: Dict[str, Any] = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in processor_inputs.items()}

    tokenizer = getattr(processor, "tokenizer", None) or getattr(
        processor, "text_tokenizer", None
    )
    if tokenizer is None:
        raise ValueError("The processor does not expose a tokenizer.")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    if is_encoder_decoder:
        with processor.as_target_tokenizer():  # type: ignore[attr-defined]
            target = processor(
                text=response_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_target_length,
            )
        labels = target["input_ids"].squeeze(0)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        features["labels"] = labels
    else:
        prompt_prefix = _format_prompt(prompt_text, response_template)
        if model_type == "open-flamingo":
            tokenized_prompt = processor(
                text=[prompt_prefix],
                images=[[image]],
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length + max_target_length,
            )
            prompt_length = tokenized_prompt["input_ids"].size(-1)
            combined = processor(
                text=[prompt_prefix + response_text],
                images=[[image]],
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length + max_target_length,
            )
            input_ids = combined["input_ids"].squeeze(0)
        else:
            prompt_encoding = processor(
                text=prompt_prefix,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length + max_target_length,
            )
            prompt_length = prompt_encoding["input_ids"].size(-1)
            combined = processor(
                text=prompt_prefix + response_text,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length + max_target_length,
            )
            input_ids = combined["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        features.update({k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in combined.items()})
        features["labels"] = labels

    return features


def prepare_datasets(
    data_args: DataArguments,
    processor: ProcessorMixin,
    model_type: str,
    is_encoder_decoder: bool,
) -> DatasetDict:
    data_files = _infer_data_files(data_args.train_file, data_args.validation_file)
    loader = _infer_dataset_loader(data_args.train_file)
    raw_datasets = load_dataset(loader, data_files=data_files)

    preprocess = partial(
        _preprocess_function,
        processor=processor,
        model_type=model_type,
        is_encoder_decoder=is_encoder_decoder,
        image_column=data_args.image_column,
        prompt_column=data_args.prompt_column,
        response_column=data_args.response_column,
        image_root=data_args.image_root,
        response_template=data_args.response_template,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
    )

    column_names = list(next(iter(raw_datasets.values())).column_names)

    with training_logger():
        processed = raw_datasets.map(
            preprocess,
            batched=False,
            num_proc=data_args.num_workers,
            remove_columns=column_names,
        )

    if data_args.max_train_samples is not None and "train" in processed:
        processed["train"] = processed["train"].select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None and "validation" in processed:
        processed["validation"] = processed["validation"].select(range(data_args.max_eval_samples))

    processed.set_format(type="torch")
    return processed


class training_logger:
    """Context manager that temporarily increases the logging verbosity."""

    def __enter__(self) -> None:
        logging.set_verbosity_info()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        logging.set_verbosity_warning()


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.set_verbosity_info()
    logger.info("Loading configuration for %s", model_args.model_id)

    config = AutoConfig.from_pretrained(
        model_args.model_id,
        revision=model_args.revision,
        cache_dir=model_args.cache_dir,
    )
    model_type = config.model_type

    processor_cls = _select_processor_class(model_type)
    processor = processor_cls.from_pretrained(
        model_args.model_id,
        revision=model_args.revision,
        cache_dir=model_args.cache_dir,
    )

    model_cls = _select_model_class(model_type)
    model = model_cls.from_pretrained(
        model_args.model_id,
        revision=model_args.revision,
        cache_dir=model_args.cache_dir,
    )

    if hasattr(model.config, "use_cache") and training_args.gradient_checkpointing:
        model.config.use_cache = False
    if training_args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = _maybe_prepare_lora(model, model_args, model_type)

    processed_datasets = prepare_datasets(
        data_args=data_args,
        processor=processor,
        model_type=model_type,
        is_encoder_decoder=config.is_encoder_decoder,
    )

    train_dataset = processed_datasets.get("train")
    eval_dataset = processed_datasets.get("validation") if "validation" in processed_datasets else None

    set_seed(training_args.seed)

    data_collator = VisionLanguageDataCollator(
        processor=processor,
        is_encoder_decoder=config.is_encoder_decoder,
        pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
    )

    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if train_dataset is not None:
        logger.info("Starting training on %d examples", len(train_dataset))
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if eval_dataset is not None:
        logger.info("Running evaluation on %d examples", len(eval_dataset))
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
