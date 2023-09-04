from copy import copy
import random
import json
import argparse
import random
from tqdm import tqdm
import numpy as np
import re
import collections
import os
from datasets import load_dataset, Dataset

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

def load_qrecc_topics(path, key='Rewrite'):
    qrecc = load_dataset('json', data_files=path)['train']

    # add unique question id
    qrecc = qrecc.map(lambda ex: {"id": f"{ex['Conversation_source']}_{ex['Conversation_no']}_{ex['Turn_no']}"})

    # combine the question and answers (ConvRerank training)
    qrecc = qrecc.map(lambda ex: {"q_and_a": f"{ex['Rewrite']} {ex['Answer']}"})

    # Get queries and search
    data_dict = {qid: query for qid, query in zip(qrecc['id'], qrecc[key])}
    return data_dict

def get_ikat_dataset(path, rewritten_path=None):
    # ikat dataset
    dataset = load_dataset('json', data_files=path)['train']
    # [TODO] Make it more easier
    if rewritten_path is not None:
        rewritten = load_topics(rewritten_path, 'generated_question')

    # flatten the turns
    data_list = []
    for topic in dataset:
        topic_id = topic['number']
        try:
            # [NOTE] `Dataset` would make the dict length consistent, 
            # so it'll add None
            ptkbs = {k: v for k, v in topic['ptkb'].items() if v is not None}
        except:
            continue

        history = []
        for turn in topic['turns']:
            data = {}

            # turn
            turn_id = turn['turn_id']

            # collect data
            data['id'] = f"{topic_id}_{turn_id}" 
            if rewritten_path is not None:
                utterance = rewritten[data['id']]
            else:
                utterance = turn['utterance']
            response = turn['response']

            ## qrecc: question / conversations/ rewrite
            data['Question'] = utterance
            data['Conversation'] = copy(history)
            data['Rewrite'] = turn['resolved_utterance']
            data['selected_ptkbs'] = [\
                    ptkbs[str(i)] for i in turn['ptkb_provenance']\
            ]

            ## use all ptkbs
            # data['all_ptkbs'] = random.sample(
            #         list(ptkbs.values()), k=len(list(ptkbs.values()))
            # )
            data['all_ptkbs'] = list(ptkbs.values())
            data_list.append(data)

            ## historical utterances
            history.append([utterance, response])

    return Dataset.from_list(data_list)

# Deprecated
# def load_ikat_topics(path, resolved=True, concat_ptkb=False):
#     data_dict = {}
#     data = json.load(open(path, 'r'))
#     for topic in data:
#         topic_id = topic['number']
#         try:
#             turns = topic['turns']
#             title = topic['title'] # more like a topic description
#             ptkbs = list(topic['ptkb'].values())
#         except:
#             # this is not the first turn of the topic
#             continue
#
#         history = []
#         for turn in turns:
#             turn_id = turn['turn_id']
#
#             utterance = ()
#             if resolved: # [Prep 1]: resolved utterances
#                 utterance += (turn['resolved_utterance'].strip(), )
#             else:        # [Prep 3]: history includes utterances & responses
#                 utterance += (" ||| ".join(history), ) 
#                 utterance += (turn['utterance'].strip(), )
#
#             if concat_ptkb: # [Prep 2]: concat ptkb
#                 selected_ptkb = turn.get('ptkb_provenance', None)
#                 if selected_ptkb:
#                     if isinstance(selected_ptkb, list):
#                         selected_ptkb = [ptkbs[p-1] for p in selected_ptkb]
#                         selected_ptkb = " ||| ".join(selected_ptkb)
#                     else:
#                         selected_ptkb = ptkbs[selected_ptkb-1]
#                 else:
#                     selected_ptkb = random.sample(ptkbs, k=1)[0]
#
#                 data_dict[f"{topic_id}_{turn_id}"] = \
#                         (selected_ptkb,) + utterance
#             else:
#                 data_dict[f"{topic_id}_{turn_id}"] = utterance
#
#             # [Prep 3]: update the historical utterances and responses
#             history.append(turn['utterance'])
#             history.append(turn['response'])
#     return data_dict

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

def load_collections(dir, full=True):
    data = {}
    for path in os.listdir(dir):
        data_subset = load_collection(os.path.join(dir, path), full=full)
        data.update(data_subset)
    return data

def load_collection(path, append=False, key='title', full=True):
    data = collections.defaultdict(str)
    fi = open(path, 'r')
    if path.endswith('tsv'):
        for i, line in enumerate(tqdm(fi)):
            if 'wiki' in path.lower():
                doc_id, content, title = line.strip().split('\t')
                content = content.strip().replace('\"', '')
                title = title.strip().replace('\"', '')
                if isinstance(append, str):
                    content = f"{title} {append} {content}"
                elif isinstance(append, tuple):
                    content = (title, content)
            else:
                doc_id, content = line.strip().split('\t')
            data[str(doc_id)] = content

            if (full is False) and (i > 10000):
                break
    else:
        for i, line in enumerate(tqdm(fi)):
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
            if (full is False) and (i > 10000):
                break
    fi.close()
    return data
