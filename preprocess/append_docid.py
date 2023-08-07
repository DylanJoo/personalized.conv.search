import os
from tqdm import tqdm 
import json
import argparse

def reversed_load_collection(path):
    data={}
    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            data[f"{item['contents']}"] = item['id']
    print('done')
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpotqa_json", default='hotpot_train_v1.1.json', type=str)
    parser.add_argument("--corpus", default='hotpot_train_v1.1.json', type=str)
    parser.add_argument("--output_qrel", default='od_hotpot_train.jsonl', type=str)
    args = parser.parse_args()

    # writer
    fout = open(args.output_qrel, 'w')

    # load data
    collection = reversed_load_collection(args.corpus)
    with open(args.hotpotqa_json, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            for i in range(len(item['evidences'])):
                evidence = item['evidences'][i]
                relevance = len(item['evidences']) - i 
                qid = item['qid']
                docid_ = f"{evidence['context']}"

                try:
                    docid = collection[docid_]
                except:
                    print('Document loss')
                    docid = 'NA'

                record = f"{qid}\t0\t{docid}\t{relevance}\n"
                fout.write(record)
