import sys
import multiprocessing
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    GenerationConfig
)
from datasets import load_dataset
from arguments import *
from data import DataCollatorForCtxRetriever

import os
os.environ["WANDB_DISABLED"] = "false"


def main():
    # Parse argument for huggingface packages
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainingArgs))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = \
                parser.parse_args_into_dataclasses()

    # Preparation 
    # (tokenizer, prompt indices)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    # Model
    from models import GTREncoder
    model = GTREncoder.from_pretrained(model_args.model_name_or_path)

    # Generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config = generation_config

    # Data
    ## Datacollator
    data_collator = DataCollatorForCtxRetriever(
            tokenizer=tokenizer, 
            max_p_length=data_args.max_p_length,
            max_q_length=data_args.max_q_length,
            truncation=True,
            padding=True,
    )

    # Data
    ## Dataset
    dataset = load_dataset('json', data_files=args.)
    n_examples = len(dataset['train'])

    # Trainer
    from trainers import TrainerForQG
    trainer = TrainerForQG(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
    )
    trainer.set_tokenizer(tokenizer)
    trainer.set_prefix(model_args.baseline_prefix)
    
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()
