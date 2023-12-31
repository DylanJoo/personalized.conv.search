import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import (
    TrainingArguments,
    Seq2SeqTrainingArguments
)

@dataclass
class ModelArgs:
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    use_auth_token: bool = field(default=False)
    temperature: Optional[float] = field(default=1)
    # for starter
    encoder_name_or_path: Optional[float] = field(default=1)
    retrieval_enhanced: bool = field(default=False)

@dataclass
class DataArgs:
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    max_p_length: int = field(default=512)
    max_q_length: int = field(default=128)
    max_src_length: int = field(default=320)
    max_tgt_length: int = field(default=32)
    precomputed_star_embeds: bool = field(default=False)

@dataclass
class TrainArgs(TrainingArguments):
    output_dir: str = field(default='./temp')
    seed: int = field(default=42)
    data_seed: int = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=-1)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=2500)
    evaluation_strategy: Optional[str] = field(default='no')
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    logging_dir: Optional[str] = field(default='./logs')
    resume_from_checkpoint: Optional[str] = field(default=None)
    save_total_limit: Optional[int] = field(default=5)
    learning_rate: Union[float] = field(default=1e-5)
    remove_unused_columns: bool = field(default=False)
    freeze_document_encoder: bool = field(default=True)
    report_to: Optional[List[str]] = field(default=None)
    alpha: Union[float] = field(default=0)
    warmup_steps: int = field(default=0)

# For starter, but it seems like there is identical to `TrainArgs`
@dataclass
class Seq2SeqTrainArgs(Seq2SeqTrainingArguments):
    output_dir: str = field(default='./temp')
    seed: int = field(default=42)
    data_seed: int = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=-1)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=2500)
    evaluation_strategy: Optional[str] = field(default='no')
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    logging_dir: Optional[str] = field(default='./logs')
    resume_from_checkpoint: Optional[str] = field(default=None)
    save_total_limit: Optional[int] = field(default=5)
    learning_rate: Union[float] = field(default=1e-5)
    remove_unused_columns: bool = field(default=False)
    report_to: Optional[List[str]] = field(default=None)
    warmup_steps: int = field(default=0)

