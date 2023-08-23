import sys
import multiprocessing
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausualLM,
    HfArgumentParser,
    GenerationConfig
)
from datasets import load_dataset

# customized modules
from data import DataCollatorForStarter
from trainers import TrainerForStarter
from models import FiDT5, GTREncoder
from arguments import ModelArgs, DataArgs, Seq2SeqTrainArgs

import os

def main():
    # Parse argument for huggingface packages
    parser = HfArgumentParser((ModelArgs, DataArgs, Seq2SeqTrainArgs))

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
    model = FiDT5.from_pretrained(model_args.model_name_or_path)
    ## precomputed statement-aware query from STatement-Aware Retriever (GTR) 
    if data_args.precomputed_star_embeds:
        star_encoder = None # None if they're precomputed
    else:
        star_encoder = GTREncoder.from_pretrained(model_args.encoder_name_or_path) 
        star_encoder.eval().cuda()

    # Generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config = generation_config

    # Data
    ## Datacollator
    data_collator = DataCollatorForStarter(
            retrieval_enhanced=model_args.retrieval_enhanced
            tokenizer=tokenizer, 
            max_p_length=data_args.max_p_length,
            max_q_length=data_args.max_q_length,
            truncation=True,
            padding=True,
            sep_token='</s>',
    )

    # Data
    ## Dataset
    dataset = load_dataset('json', data_files=data_args.train_file)
    n_examples = len(dataset['train'])
    if training_args.do_eval:
        dataset = dataset['train'].train_test_split(test_size=100, seed=1997)

    # Trainer
    trainer = TrainerForStarter(
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
