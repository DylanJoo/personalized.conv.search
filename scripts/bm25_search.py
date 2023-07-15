import os
from tqdm import tqdm 
import json
import argparse
from pyserini.search.lucene import LuceneSearcher
from utils import load_collection, load_query

def search(args):
    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(k1=args.k1, b=args.b)

    query = load_query(args.query)

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
    parser.add_argument("--k1",type=float, default=0.82) 
    parser.add_argument("--b", type=float, default=0.68) 
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--output", default='runs/run.sample.txt', type=str)
    parser.add_argument("--query", default=None, type=str)
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)
    search(args)
    print("Done")
