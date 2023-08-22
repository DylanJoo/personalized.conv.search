import re
from tqdm import tqdm 
import json
import argparse
from transformers import AutoModelForSequenceClassification
from tool import load_collection, load_qrecc_topics, load_runs

def extract_statement(texts):
    texts = re.sub("[1]\;", "", texts)
    texts = re.sub("[Bb]ackground [0-9]*\:", ";", texts)
    statements = texts.split(';')
    statements = [s.strip() for s in statements if len(s) >= 10]
    return statements

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default='wiki_psgs_w100.tsv')
    parser.add_argument("--run", default='qrecc.run')
    parser.add_argument("--topic", default='qrecc_train.json')
    parser.add_argument("--input_jsonl", default='input.jsonl')
    parser.add_argument("--output_jsonl", default='output.jsonl')

    # filtering
    ## [NOTE] discard the statement filtering for starter
    args = parser.parse_args()

    # load data
    collection = load_collection(args.collection, append=())
    run = load_runs(args.run)
    answers = load_qrecc_topics(args.topic, 'Answer')
    fout = open(args.output_jsonl, 'w') 

    # load statements
    with open(args.input_jsonl, 'r') as fin:
        for line in tqdm(fin):
            item = json.loads(line.strip())
            qid = item['qid']
            query = item['query']

            # get topic and pseudo-relevant contexts
            docids = run[qid]
            contexts = [tuple(collection[docid]) for docid in docids]
            answer = answers[qid]

            # statements
            # [NOTE] we only use first retrieved passage as context for llm
            statements = extract_statement(item['llm_generated_texts'])

            fout.write(json.dumps({
                "qid": qid,
                "question": query,
                "docids": docids,
                "contexts": contexts, 
                "answer": answer,
                "statements": statements,
            }, ensure_ascii=False)+'\n')

    print("Done")
