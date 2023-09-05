import json
from tool import load_collections
data = []

clueweb = load_collections('/tmp2/trec/ikat/data/collection/ikat/')

with open('../runs/ikat.test.bm25.top1000.run', 'r') as f:
    for line in f:
        docid = line.strip().split()[2]
        data.append(docid)

data = set(data)
with open('../data/ikat/clueweb.jsonl', 'w') as f:
    for docid in data:
        f.write(json.dumps({"id": docid, "contents": clueweb[docid]}, ensure_ascii=False)+'\n')
