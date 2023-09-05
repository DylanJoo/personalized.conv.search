import os
from tqdm import tqdm 
import json
import argparse
from tool import (
    load_runs,
    load_topics,
    load_collection,
    get_ikat_dataset
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_clueweb", default=None, type=str)
    parser.add_argument("--list_wiki", default=None, type=str)
    parser.add_argument("--collection_clueweb", default=None, type=str)
    parser.add_argument("--collection_wiki", default=None, type=str)
    parser.add_argument("--submission_file", action=None, type=str)
    parser.add_argument("--submission_name", action=None, type=str)
    parser.add_argument("--provenance_file", action=None, type=str)
    # add dataset
    parser.add_argument("--ikat_file", action=None, type=str)
    # add rewritten
    parser.add_argument("--rewritten_file", action=None, type=str)
    # add reseponse
    parser.add_argument("--response_file", action=None, type=str)
    args = parser.parse_args()

    run_clueweb = load_runs(args.list_clueweb, output_score=True)
    run_wiki = load_runs(args.list_wiki)

    try:
        clueweb = load_collection(args.collection_clueweb)
    except:
        clueweb = None
    wiki = load_collection(args.collection_wiki, append=())

    ikat = get_ikat_dataset(args.ikat_file, args.rewritten_file)
    query = {qid: text for qid, text in zip(ikat['id'], ikat['Question'])}
    statements = {qid: text for qid, text in zip(ikat['id'], ikat['all_ptkbs'])}

    if args.response_file:
        response = load_topics(args.response_file, 'response')

    submission = {}
    submission['run_name'] = args.submission_name
    submission['run_type'] = 'automatic'
    submission['turns'] = []

    # (1) preapre submission template
    with open(args.provenance_file, 'w') as fp:
        for qid in run_clueweb:
            # provenance
            ranklist_clueweb = run_clueweb[qid][:100]
            ranklist_wiki = run_wiki[qid]

            turn = {} 
            turn['turn_id'] = qid
            turn['responses'] = []

            # add a response
            turn['responses'] = [{
                'rank': 1,
                'text': response[qid] if args.response_file is not None else "NA",
                'ptkb_provenance': [],
                'passage_provenance': []
            }]

            # only one generation
            for i, docid_score in enumerate(ranklist_clueweb):
                docid, score = docid_score

                turn['responses'][0]['passage_provenance'].append({
                    "id": docid, 
                    "text": "...",
                    "score": score,
                    "used": 'true' if i <= 10 else 'false'
                })

            # append to submission
            submission['turns'].append(turn)

            ## (2) prepare generation provenances
            ### use 10 for each
            context = [wiki[docid] for docid in ranklist_wiki[:10]]
            try:
                context += [("<pad>", clueweb[docid]) for docid, _ in ranklist_clueweb[:10]]
            except:
                print('no clueweb')

            fp.write(json.dumps({
                "qid": qid, 
                "question": query[qid],
                "context": context,
                "statements": statements[qid]
            }, ensure_ascii=False)+'\n')

        # writer
        json.dump(submission, open(args.submission_file, 'w'), indent=2)
