import re
from tqdm import tqdm 
import json
import argparse
from transformers import AutoModelForSequenceClassification
from tool import load_collection

def extract_statement(texts):
    texts = re.sub("[1]\;", "", texts)
    statements = texts.split(';')
    statements = [s.strip() for s in statements if len(s) >= 10]
    return statements

def consistency_filter(statements, query, 
                       model, tokenizer, topk=-1, threshold=0):
    """ this stage aims to filter the high-quality statements; 
    such statements may include the relevant one or non-relevant one.

    Method:
    [1] leverage the NLI model to achieve.
    [2] ...
    """
    inputs = tokenizer(
            [f"Query: {query} Context: {s}" for s in statements],
            pading=True,
            max_length=64,
            truncation=True,
            padding=True,
            return_tensors='pt'
    ).to(model.device)

    # [TODO] include the post-model processing if needed. 
    scores = model(**inputs).detach().cpu().numpy()
    outputs = sorted(
            list(zip(statements, scores)), key=lambda x: x[1], reverse=True
    )
    if topk:
        threshold = max(sorted(scores)[topk], threshold)
    return [text for (text, score) in outputs if score > threshold]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default='wiki_psgs_w100.tsv')
    parser.add_argument("--input_jsonl", default='input.jsonl')
    parser.add_argument("--output_jsonl", default='output.jsonl')

    # filtering
    parser.add_argument("--filtering", default=False, action='store_true')
    parser.add_argument("--filter_model", default=None, type=str)
    parser.add_argument("--filter_k", default=10, type=int)
    parser.add_argument("--filter_thres", default=0.9, type=float)
    args = parser.parse_args()

    # load data
    collection = load_collection(args.collection, full=False)
    fout = open(args.output_jsonl, 'w') 

    # filtering
    if args.filtering:
        model = AutoModelForSequenceClassification.from_pretrained(args.filter_model)
        tokenizer = AutoTokenizer.from_pretrained(args.filter_model)

    with open(args.input_jsonl, 'r') as fin:
        for line in tqdm(fin):
            item = json.loads(line.strip())
            qid = item['qid']
            query = item['query']
            document = collection[item['docid']]

            # Generated statements
            ## extract 
            statements = extract_statement(item['llm_generated_texts'])

            ## filter
            if args.filtering:
                statements = consistency_filter(
                        statements, query, 
                        model, tokenizer, args.filter_k, args.filter_thres
                )

            for statement in statements:
                fout.write(json.dumps({
                    "qid": qid,
                    "query": query,
                    "statement": statement,
                    "passage": document,
                }, ensure_ascii=False)+'\n')

    print("Done")
