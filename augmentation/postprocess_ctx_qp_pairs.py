import re
from tqdm import tqdm 
import json
import argparse
from tool import load_collection

def extract_statement(text):
    pass

def consistency_filter(statements, document, model_name_or_path):
    texts = [f"Query: {} {}"]
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default='input.jsonl')
    parser.add_argument("--output_jsonl", default='output.jsonl')

    # filtering
    parser.add_argument("--filtering", default=False, action='store_true')
    parser.add_argument("--filter_k", default=10, type=int)
    parser.add_argument("--filter_thres", default=float, type=int)
    args = parser.parse_args()

    # load data
    collection = load_collection()
    fout = open(args.output_jsonl, 'w') 

    with open(args.input_jsonl, 'r') as fin:
        for line in tqdm(fin):
            item = json.loads(line.strip())
            query = item['query']
            document = collection[item['pid']]

            # Generated statements
            ## extract 
            statements = extract_statement(item['llm_generated_text'])
            ## filter
            if args.filtering:
                statements = consistency_filter(statements)
            ## write
            for statement in statements:
                fout.write(json.dumps({
                    "query": 0,
                    "statement": 1,
                    "passage": 2,
                }, ensure_ascii=False)+'\n')
    print("Done")
