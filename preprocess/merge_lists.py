import os
from tqdm import tqdm 
import json
import argparse
from tool import load_runs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_clueweb", default=None, type=str)
    parser.add_argument("--list_wiki", default=None, type=str)
    parser.add_argument("--submission_file", action=None, type=str)
    parser.add_argument("--submission_name", action=None, type=str)
    parser.add_argument("--provenance_file", action=None, type=str)
    args = parser.parse_args()

    run_clueweb = load_runs(args.list_clueweb, output_score=True)
    # run_wiki = load_runs(args.list_wiki)

    submission = {}
    submission['run_name'] = args.submission_name
    submission['run_type'] = 'automatic'
    submission['turns'] = []

    # (1) preapre submission template
    with open(args.provenance_file, 'w') as fp:
        for qid in run_clueweb:
            # provenance
            ranklist_clueweb = run_clueweb[qid][:1000]
            # ranklist_wiki = run_wiki[qid]

            turn = {}
            turn['turn_ids'] = qid
            turn['responses'] = []

            # add a response
            turn['responses'] = [{
                'rank': 1,
                'text': None,
                'ptkb_provenance': [],
                'passage_provenance': []
            }]

            # only one generation
            for i, docid_score in enumerate(ranklist_clueweb):
                docid, score = docid_score

                turn['responses'][0]['passage_provenance'].append({
                    "id": docid, 
                    "text": "",
                    "score": score,
                    "used": 'true' if i <= 10 else 'false'
                })

            # append to submission
            submission['turns'].append(turn)

        # writer
        json.dump(submission, open(args.submission_file, 'w'), indent=2)
