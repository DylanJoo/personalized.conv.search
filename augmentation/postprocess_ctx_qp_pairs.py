import re
from tqdm import tqdm 
import json
import argparse
from tool import load_collection

def extract_statement(texts):
    texts = re.sub("[1]\;", "", texts)
    statements = texts.split(';')
    statements = [s.strip() for s in statements if len(s) >= 10]
    return statements

def consistency_filter(statements, document, model_name_or_path):
    """ this stage aims to filter the high-quality statements; 
    such statements may include the relevant one or non-relevant one.

    Method:
    [1] leverage the NLI model to achieve.
    [2] ...
    """
    texts = [f"Query: {} {}"]
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default='input.jsonl')
    parser.add_argument("--output_jsonl", default='output.jsonl')

    # filtering
    parser.add_argument("--filtering", default=False, action='store_true')
    parser.add_argument("--filter_model", default=None, type=str)
    parser.add_argument("--filter_k", default=10, type=int)
    parser.add_argument("--filter_thres", default=0.9, type=float)
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
                statements = consistency_filter(query, statements)

            for statement in statements:
                fout.write(json.dumps({
                    "query": query,
                    "statement": statement,
                    "passage": document,
                }, ensure_ascii=False)+'\n')

    print("Done")
