import os
from tqdm import tqdm 
import json
import argparse
from pyserini.search.lucene import LuceneSearcher
from tool import get_ikat_dataset
from datasets import load_dataset

def search(index_dir, dataset, output='sample.trec', k=1000, k1=0.9, b=0.4):
    # init lucene searcher
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=k1, b=b)

    # search for each q
    for query in tqdm(dataset, total=len(dataset)):
        qid, qtext = query['id'], query['Question']
        hits = searcher.search(qtext, k=k)
        for i in range(len(hits)):
            output.write(f'{qid} Q0 {hits[i].docid:4} {i+1} {hits[i].score:.5f} BM25\n')

    output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=100, type=int)
    parser.add_argument("--k1",type=float, default=0.9) 
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--output", default='runs/run.sample.txt', type=str)
    parser.add_argument("--topic", default=None, type=str)
    parser.add_argument("--concat_ptkb", default=False, action='store_true')
    parser.add_argument("--rewritten", default=None, type=str)
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)

    # dataset
    ## load from json
    dataset = get_ikat_dataset(args.topic)

    ## preprocess 
    if args.rewritten is not None:
        rewritten = load_dataset(
                'json', data_files=args.rewritten, keep_in_memory=True
        )['train']
        rewritten = {k: v for k, v in zip(rewritten['qid'], rewritten['generated_question'])}
        dataset.map(lambda x: {'Question': rewritten[x['id']]})

    if args.concat_ptkb:
        # all the ptkbs 
        dataset = dataset.map(lambda x: {'Question': " | ".join(x['all_ptkbs']) + x['Question']})

    # output writier
    writer = open(args.output, 'w')

    # search
    search(
            index_dir=args.index,
            dataset=dataset,
            output=writer,
            k=args.k, k1=args.k1, b=args.b
    )
    print("Done")
