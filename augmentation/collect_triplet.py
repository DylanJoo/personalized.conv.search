import argparse
import os
from tqdm import tqdm 
import json
import argparse
import sys
from tool import load_collection, load_runs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--flatten_qa", default='flatten_hotpotqa.json', type=str)
    parser.add_argument("--run", default='hotpotqa.train.contriever.run', type=str)
    parser.add_argument("--corpus", default='passages.json', type=str)
    parser.add_argument("--triplet", default='prediction.jsonl', type=str)
    args = parser.parse_args()

    # writer
    fout = open(args.triplet, 'w')

    # load data
    run = load_runs(args.run)

    with open(args.flatten_qa, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            qid = item['qid']
            question = item['question']
            answer = item['answer']

            # positive from supporting document in hotpotqa
            positives = []
            for evidence in item['evidences']:
                try:
                    positives.append(evidence['docid'])
                except:
                    print("no positive supporting doc")

            # hard negative from contriever
            if len(positives) > 0: # sometimes, there is no positive docid
                negatives = []
                for docid_candidate in run[qid][10:]: 
                    if docid_candidate not in positives:
                        negatives.append(docid_candidate)
                    if len(negatives)>=10:
                        break
                # write out
                fout.write(json.dumps(
                    {'qid': qid, 'positive_docids': positives, 'negative_docids': negatives}
                )+'\n')
