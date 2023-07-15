import os
from tqdm import tqdm 
import json
import argparse
from pyserini.search.lucene import LuceneSearcher

def search(args):
    # init lucene searcher
    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(k1=args.k1, b=args.b)

    # load topic (using manual rewritten question)
    query = {}
    data = json.load(open(args.topic, 'r'))
    for topic in data:
        topic_id = topic['number']
        turns = topic['turn']

        for turn in turns:
            turn_id = turn['number']
            question = turn['manual_rewritten_utterance']
            query[f"{topic_id}_{turn_id}"] = question.strip()
            # utterance = turn['raw_utterance']
            # passage = turn['passage']

    # prepare the output file
    output = open(args.output, 'w')

    # search for each q
    for qid, qtext in tqdm(query.items()):
        hits = searcher.search(qtext, k=args.k)
        for i in range(len(hits)):
            output.write(f'{qid} Q0 {hits[i].docid:4} {i+1} {hits[i].score:.5f} bm25\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--k1",type=float, default=4.68) # 0.5 # 0.82
    parser.add_argument("--b", type=float, default=0.87) # 0.3 # 0.68
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--output", default='runs/run.sample.txt', type=str)
    parser.add_argument("--topic", default=None, type=str)
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)
    search(args)
    print("Done")
