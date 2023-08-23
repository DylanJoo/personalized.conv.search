"""
Before running this code, you should get the `starter` raw dataset.
Check `augmentation/convert_llm_to_starter_train.py` for detail.
"""
import json
import argparse
import collections
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from models import GTREncoder
from transformers import AutoTokenizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--input_jsonl", type=str, default='test.jsonl')
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--max_num_statements", type=int, default=10)
    parser.add_argument("--min_num_statements", type=int, default=5)
    parser.add_argument("--sep_token", type=str, default='</s>')
    parser.add_argument("--device", type=str, default='0')
    args = parser.parse_args()

    # load model
    encoder = GTREncoder.from_pretrained(args.model_name_or_path)
    encoder.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    fout = open(args.output_jsonl, 'w')

    # load data
    with open(args.input_jsonl, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            question = item['question']
            statements = item['statements']
            template = "{0} {1} {2}"

            if len(statements) >= args.min_num_statements:

                ## enumerate statements (at least n_min, at most n_max)
                starter_texts = []
                for statement in statements[:args.max_num_statements]:
                    starter_texts.append(template.format(
                        question, args.sep_token, statement
                    ))
                ## add the dummy statements 
                for i in range(max(1, 1+args.max_num_statements-len(statements))):
                    starter_texts.append(template.format(
                        question, args.sep_token, ""
                    ))
                print('2', len(starter_texts))

                ## get embeddings
                ### tokenizaetion
                tokenizer_inputs = tokenizer(
                        starter_texts, 
                        max_length=args.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                ).to(args.device)

                ### encoding
                embeddings = encoder.encode(
                        tokenizer_inputs, normalized=False, projected=False
                )
                embeddings = embeddings.detach().cpu().numpy().tolist()

                ### writer
                item.update({"statament_aware_embeds": embeddings})
                fout.write(json.dumps(item, ensure_ascii=False)+'\n')

    fout.close()
