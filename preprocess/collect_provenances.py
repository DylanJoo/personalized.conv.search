import os
from tqdm import tqdm 
import json
import argparse
from glob import glob

def collect(passage_dir, output):
    fout = open(output, 'w')

    # load collections
    passages = {}
    for file in glob(passage_dir+"*psg*.jsonl"):
        with open(file, 'r') as f:
            for line in f:
                data=json.loads(line.strip())
                psgid = f"{data['doc_id']}:{data['passage_id']}"
                content = data['passage_text']
                content = content.replace('\\', '')
                content = content.replace('\n', '')

                passages[psgid] = content
                fout.write(json.dumps({
                    "id": psgid, 
                    "contents": content
                }, ensure_ascii=True)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--passage_dir", default='./', type=str)
    parser.add_argument("--output", default='passages.jsonl', type=str)
    args = parser.parse_args()

    collect(args.passage_dir, args.output)
    print("Done")
