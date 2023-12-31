import numpy as np
import os
import torch
from tqdm import tqdm 
import json
import argparse
from tool import (
    load_collection,
    batch_iterator, 
    get_ikat_dataset
)
from contriever import (
    ContrieverQueryEncoder, 
    ContrieverDocumentEncoder
)
from collections import defaultdict

def rerank_ptkbs(star_queries, docs, q_encoder, d_encoder):
    d_embs = d_encoder.encode(docs)
    scores = []
    for i, star_query in enumerate(star_queries):
        query = star_query[0]
        q_embs = np.array([
            q_encoder.encode(f"{query} [SEP] {statement}") for statement in star_query[1]
        ])
        scores.append(np.mean(q_embs * d_embs[i], 0).sum())
    return np.array(scores).flatten().tolist()

def rerank(queries, docs, q_encoder, d_encoder):
    q_embs = np.array([q_encoder.encode(q) for q in queries])
    d_embs = d_encoder.encode(docs)
    scores = np.matmul(q_embs, d_embs.T).diagonal()
    return scores.tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_encoder_path", default=None, type=str)
    parser.add_argument("--d_encoder_path", default=None, type=str)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    # special args for ikat
    parser.add_argument("--run", default=None, type=str)
    parser.add_argument("--query", default=None, type=str)
    parser.add_argument("--collection", default=None, type=str)
    parser.add_argument("--output_run", default=None, type=str)
    parser.add_argument("--rewritten", default=None, type=str)
    parser.add_argument("--concat_ptkb", default=False, action='store_true')
    args = parser.parse_args()

    # load bi-encoder
    query_encoder = ContrieverQueryEncoder(args.q_encoder_path, device=args.device)
    doc_encoder = ContrieverDocumentEncoder(args.d_encoder_path, device=args.device)

    # load data
    dataset = get_ikat_dataset(args.query, rewritten_path=args.rewritten)
    ## rewritten or utterance
    if args.concat_ptkb:
        query = {x['id']: (x['Question'], x['all_ptkbs']) for x in dataset}
    else:
        query = {x['id']: x['Question'] for x in dataset}
    collections = load_collection(args.collection)
    run_lines = open(args.run).readlines()
    ranking_list = defaultdict(list)

    for lines in tqdm(
            batch_iterator(run_lines, args.batch_size),
            total=(len(run_lines)//args.batch_size)+1
        ):
        # load runs with lines
        lines = [line.strip().split() for line in lines]
        qids = [line[0] for line in lines]
        docids = [line[2] for line in lines]
        queries = [query[qid] for qid in qids]
        documents = [collections[docid] for docid in docids]

        # rerank
        if args.concat_ptkb:
            rel_scores = rerank_ptkbs(queries, documents, query_encoder, doc_encoder)
        else:
            rel_scores = rerank(queries, documents, query_encoder, doc_encoder)

        for qid, docid, score in zip(qids, docids, rel_scores):
            ranking_list[qid].append( (docid, float(score)) )

    # write output
    fout = open(args.output_run, 'w')
    for qid, candidate_passage_list in ranking_list.items():
        candidate_passage_list = sorted(
                candidate_passage_list, key=lambda x: x[1], reverse=True
        )

        for idx, (docid, score) in enumerate(candidate_passage_list):
            example = f'{qid} Q0 {docid} {str(idx+1)} {score} BM25-DR\n'
            fout.write(example)

    fout.close()
