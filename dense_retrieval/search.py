import os
import torch
from tqdm import tqdm 
import json
import argparse
from pyserini.search import FaissSearcher
from tool import (
    load_collection, 
    batch_iterator, 
    load_topics, 
    load_qrecc_topics,
    get_ikat_dataset
)
from contriever import ContrieverQueryEncoder
from gtr import GTREncoder

def search(args, rewritten=None, concat_ptkb=False):

    if 'contriever' in args.encoder_path:
        query_encoder = ContrieverQueryEncoder(args.encoder_path, args.device)
        searcher = FaissSearcher(args.index, query_encoder)
    elif 'gtr' in args.encoder_path:
        query_encoder = GTREncoder(args.encoder_path, args.device)
        searcher = FaissSearcher(args.index, query_encoder)
    else:
        searcher = FaissSearcher(args.index, args.encoder_path)

    if torch.cuda.is_available():
        searcher.query_encoder.model.to(args.device)
        searcher.query_encoder.device = args.device

    # for example
    if "ikat" in args.query.lower():
        query = get_ikat_dataset(args.query, rewritten_path=args.rewritten)
        if args.concat_ptkb:
            qids, qtexts = [], []
            for i in range(len(query)):
                for j, ptkb in enumerate(query['all_ptkbs'][i]):
                    qids.append(f"{query['id'][i]}:{j}")
                    qtexts.append(f"{query['Question'][i]} [SEP] {ptkb}")
        else:
            qids = query['id']
            qtexts = query['Question']

    elif 'qrecc' in args.query.lower():
        query = load_qrecc_topics(args.query)
        qids = list(query.keys())
        qtexts = list(query.values())
    else:
        query = load_topics(args.query)
        qids = list(query.keys())
        qtexts = list(query.values())

    # prepare the output file
    output = open(args.output, 'w')

    # search for each q
    if args.batch_size == 1:
        for qid, qtext in zip(qids, qtexts):
            hits = searcher.search(qtext, k=args.k, threads=10)
            for i in range(len(hits)):
                output.write(f'{qid} Q0 {hits[i].docid:4} {i+1} {hits[i].score:.5f} FAISS\n')
    else:
        for (start, end) in tqdm(
                batch_iterator(range(0, len(qids)), args.batch_size, return_index=True),
                total=(len(qids)//args.batch_size)+1
            ):
            qids_batch = qids[start: end]
            qtexts_batch = qtexts[start: end]
            hits = searcher.batch_search(
                    queries=qtexts_batch, 
                    q_ids=qids_batch, 
                    k=args.k
            )
            for key, value in hits.items():
                for i in range(len(hits[key])):
                    output.write(f'{key} Q0 {hits[key][i].docid:4} {i+1} {hits[key][i].score:.5f} FAISS\n')
    output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--output", default='runs/run.sample.txt', type=str)
    parser.add_argument("--encoder_path", default=None, type=str)
    # special args
    parser.add_argument("--query", default=None, type=str)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    # special args for ikat
    parser.add_argument("--rewritten", default=None, type=str)
    parser.add_argument("--concat_ptkb", default=False, action='store_true')
    args = parser.parse_args()

    os.makedirs('runs', exist_ok=True)
    search(args, args.rewritten, args.concat_ptkb)
    print("Done")
