import json
import argparse
import random
from tqdm import tqdm
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

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def load_topics(path, selected_key='question'):
    data_dict = {}
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            idname = [key for key in data if 'id' in key][0]
            keyname = selected_key
            data_dict[data.pop(idname)] = data[keyname]
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

def load_collection(path, append=False, key='title'):
    data = collections.defaultdict(str)
    fi = open(path, 'r')
    for line in tqdm(fi):
        item = json.loads(line.strip())
        doc_id = item.pop('id')
        if append:
            title = item['title']
            content = item['contents']
            data[str(doc_id)] = f"{title}{append}{content}"
        else:
            if 'contents' in item:
                key = 'contents'
            data[str(doc_id)] = item[key]
    return data
