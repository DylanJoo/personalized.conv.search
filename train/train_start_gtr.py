import sys
import multiprocessing
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig
)
from datasets import load_dataset

# customized modules
from data import DataCollatorForCtxRetriever
from models import GTREncoder
from trainers import TrainerForStart
from arguments import ModelArgs, DataArgs, TrainArgs

import os

def main():
    # Parse argument for huggingface packages
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))

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
    model = GTREncoder.from_pretrained(model_args.model_name_or_path)
    ## freezing docs: see if it wil be faster by loading from index
    if training_args.freeze_document_encoder:
        model_freezed = GTREncoder.from_pretrained(model_args.model_name_or_path) 
        model_freezed.eval() 
        model_freezed.cuda()
    else:
        model_freezed = None

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
    dataset = load_dataset('json', data_files=data_args.train_file)
    n_examples = len(dataset['train'])
    if training_args.do_eval:
        dataset = dataset['train'].train_test_split(test_size=100, seed=1997)

    # Trainer
    trainer = TrainerForStart(
            document_encoder=model_freezed,
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
    )
    
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()
