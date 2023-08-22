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
            starter_texts = []
            item = json.loads(line.strip())
            question = item['question']
            statements = item['statements']
            template = "{0} {1} {2}"

            ## enumerate statements
            for statement in statements:
                starter_texts.append(
                        template.format(question, args.sep_token, statement)
                )
                # print(template.format(question, args.sep_token, statement))

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
            item.update({"past_key_values": embeddings})
            fout.write(json.dumps(item, ensure_ascii=False)+'\n')

    fout.close()
