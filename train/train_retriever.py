import os
import sys
from typing import Optional, Union
from transformers import (
    HfArgumentParser,
    DataCollatorForSeq2Seq
)
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoConfig
)
from datasets import load_dataset
from arguments import ModelArgs, DataArgs, TrainArgs

os.environ["WANDB_DISABLED"] = "false"


def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Config and Tokenizer 
    config = AutoConfig.from_pretrained(model_args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    # Model
    # backbone and pretrained 
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    
    # Data
    # collator/preprocessor
    from datacollator import DataCollatorForDesc2Title, DataCollatorForProduct2Query
    datacollator_class_map = {
            "desc2title": DataCollatorForDesc2Title,
            "product2query": DataCollatorForProduct2Query
    }
    for key in datacollator_class_map:
        if key in training_args.output_dir:
            data_collator = datacollator_class_map[key](
                    tokenizer=tokenizer,
                    max_src_length=data_args.max_src_length,
                    max_tgt_length=data_args.max_tgt_length,
            )

    # Data: dataset
    dataset = load_dataset(
            'json', data_files=data_args.train_file
    )['train'].train_test_split(test_size=0.0001)
    dataset_train = dataset['train']
    dataset_eval = dataset['test']

    trainer = Seq2SeqTrainer(
            model=model, 
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_eval,
            data_collator=data_collator,
    )
    
    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()
