"""
Before running this code, you should get the `starter` raw dataset.
Check `augmentation/convert_llm_to_starter_train.py` for detail.
"""
import json
import torch
import argparse
import collections
import numpy as np
from tqdm import tqdm
# customized modules
from data import DataCollatorForStarter
from models import FiDT5
from transformers import AutoTokenizer
from tool import get_ikat_dataset, load_topics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--input_jsonl", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--rewritten", type=str, default=None)
    parser.add_argument("--device", type=str, default='0')
    args = parser.parse_args()

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = FiDT5.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    fout = open(args.output_jsonl, 'w')

    with open(args.input_jsonl, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            question = item['question']
            context = item['context']
            N = len(context)
            template = "question: {0} title: {1} passage: {2}"

            ## get embeddings
            ### tokenizaetion
            inputs = tokenizer(
                    [template.format(question, *c) for c in context], 
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
            )

            inputs['input_ids'] = inputs['input_ids'].view(
                    -1, N, inputs['input_ids'].size(-1)
            )
            inputs['attention_mask'] = inputs['attention_mask'].view(
                    -1, N*inputs['attention_mask'].size(-1)
            )

            # retrieval enhanced
            if 'statement_aware_embeds' in item:
                inputs['past_key_values'] = torch.stack([
                    torch.tensor(item['statement_aware_embeds'])
                ], dim=0)

            inputs = inputs.to(args.device)
            ### generate
            outputs = model.generate(**inputs, num_beams=5, max_new_tokens=384)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            fout.write(json.dumps(
                {"qid": item['qid'], "response": response}
            , ensure_ascii=False)+'\n')

