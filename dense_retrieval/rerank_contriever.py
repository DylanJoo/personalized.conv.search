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
from contriever import ContrieverQueryEncoder, DocumentEncoder

def rerank(args):

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_encoder_path", default=None, type=str)
    parser.add_argument("--d_encoder_path", default=None, type=str)
    parser.add_argument("--output", default='runs/run.sample.txt', type=str)
    # special args
    parser.add_argument("--query", default=None, type=str)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    # special args for ikat
    parser.add_argument("--rewritten", default=None, type=str)
    parser.add_argument("--concat_ptkb", default=False, action='store_true')
    args = parser.parse_args()

    # load bi-encoder
    query_encoder = ContrieverQueryEncoder(args.q_encoder_path, args.device)
    doc_encoder = ContrieverDocumentEncoder(args.d_encoder_path, args.device)

    # [NOTE] So far, no query encoder
    elif 'gtr' in args.encoder_path:
        query_encoder = GTREncoder(args.encoder_path, args.device)

    # load data
    queries = load_query()
    if (args.description is True) and (args.title is True):
        collection = load_collection(args.collection, append=' ', key=None)
        print('used both title and description')
    elif args.title is True:
        collection = load_collection(args.collection, append=False, key='title')
        print('used only title')
    elif args.description is True:
        collection = load_collection(args.collection, append=False, key='description')
        print('used both description')

    # prepare data for monot5
    qp_pairs = load_qp_pair(args.run, topk=args.topk)
    data = Dataset.from_dict(qp_pairs)
    data = data.map(lambda x: monot5_preprocess(x, queries, collection))

    # data loader
    datacollator = DataCollatorForCrossEncoder(
            tokenizer=model.tokenizer,
            padding=True,
            max_length=args.max_length,
            truncation=True,
    )
    dataloader = DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=datacollator
    )

    model.to(args.device)
    model.eval()

    ranking_list = collections.defaultdict(list)

    # run prediction
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch_inputs, pair_id = batch
        output = model.predict(batch_inputs)

        true_prob = output[:, 0]
        # false_prob = output[:, 1]

        for t_prob, (qid, docid) in zip(true_prob, pair_id):
            ranking_list[qid].append((docid, t_prob))

    # write output
    fout = open(args.output_trec, 'w')
    for qid, candidate_passage_list in ranking_list.items():
        # Using true prob as score, so reverse the order.
        candidate_passage_list = sorted(candidate_passage_list, key=lambda x: x[1], reverse=True)

        for idx, (docid, t_prob) in enumerate(candidate_passage_list[:1000]):
            example = f'{qid} Q0 {docid} {str(idx+1)} {t_prob} {args.prefix}\n'
            fout.write(example)

    fout.close()
