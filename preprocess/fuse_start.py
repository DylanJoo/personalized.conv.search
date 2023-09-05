import os
from tqdm import tqdm 
import json
import argparse
from tool import load_runs
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_run", default=None, type=str)
    parser.add_argument("--output_run", action=None, type=str)
    args = parser.parse_args()

    run = load_runs(args.input_run)
    ranklist = defaultdict(list)

    with open(args.output_run, 'w') as fp:

        # collect qid ptkb results
        for qid_ptkb in run:
            qid, ptkb = qid_ptkb.split(":")

            for i, docid in enumerate(run[qid_ptkb]):
                ranklist[qid].append( (docid, 1/(i+1)) )

        # fuse results
        for qid in ranklist:
            temp = defaultdict(float)

            for docid, score in ranklist[qid]:
                temp[docid] += score

            # sort
            ranklist_fused = sorted(
                    [(docid, fusion) for docid, fusion in temp.items()], 
                    key=lambda x: x[1], reverse=True
            )

            for i, (docid, score) in enumerate(ranklist_fused):
                fp.write(f"{qid} Q0 {docid} {i+1} {score} START-RRF\n")
