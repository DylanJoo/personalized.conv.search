import json
import argparse
import random
import numpy as np
import re
import collections
import os

def normalized(x):
    x = x.strip()
    x = x.replace("\t", " ")
    x = x.replace("\n", " ")
    x = re.sub("\s\s+" , " ", x)
    return x

def load_queries(path):
    query_dict = {}
    with open(path, 'r') as f:
        for line in f:
            qid, qtext = line.strip().split('\t')
            query_dict[qid] = qtext.strip()
    return query_dict

def load_topics(path):
    data_dict = {}
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            idname = [key for key in data if 'id' in key][0]
            data_dict[data.pop(idname)] = data
    return data_dict

def load_runs(path, output_score=False): # support .trec file only
    run_dict = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            run_dict[qid] += [(docid, float(rank), float(score))]

    sorted_run_dict = collections.OrderedDict()
    for (qid, doc_id_ranks) in run_dict.items():
        sorted_doc_id_ranks = \
                sorted(doc_id_ranks, key=lambda x: x[1], reverse=False) # score with descending order
        if output_score:
            sorted_run_dict[qid] = [(docid, rel_score) for docid, rel_rank, rel_score in sorted_doc_id_ranks]
        else:
            sorted_run_dict[qid] = [docid for docid, _, _ in sorted_doc_id_ranks]

    return sorted_run_dict

def load_collections(path=None, dir=None, candidate_set=None):
    collection_dict = {}

    if dir: # load if there are many jsonl files
        files = [os.path.join(dir, f) for f in os.listdir(dir) if ".json" in f]
    else:
        files = [path]

    for file in files:
        print(f"Loading from collection {file}...")
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                example = json.loads(line.strip())
                if candidate_set:
                    if example['id'] in candidate_set:
                        collection_dict[example['id']] = example['contents'].strip()
                        candidate_set.remove(example['id'])
                    if len(candidate_set) == 0:
                        break
                else:
                    try:
                        collection_dict[example['id']] = example['contents'].strip()
                    except:
                        collection_dict[example['id']] = example['contents']

                if i % 1000000 == 1:
                    print(f" # documents...{i}")

    print("DONE")
    return collection_dict