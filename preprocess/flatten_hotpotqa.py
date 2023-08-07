import os
from tqdm import tqdm 
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpotqa_json", default='hotpot_train_v1.1.json', type=str)
    parser.add_argument("--output", default='flatten_hotpotqa.jsonl', type=str)
    parser.add_argument("--return_corpus", action='store_true', default=False)
    args = parser.parse_args()

    # load data
    data = json.load(open(args.hotpotqa_json, 'r')) 
    corpus = []
    corpus_titles = list()

    # writer
    fout = open(args.output, 'w')

    if args.return_corpus:
        fcorpus = open(args.output.rsplit('/', 1)[0] + "/passages.jsonl" ,'w')

    for i in tqdm(range(len(data))):
        facts = data[i]['supporting_facts']
        facts = [(f[0], f[1]) for f in facts]
        query = data[i]['question']
        answer = data[i]['answer']
        qid = data[i]['_id']
        evidences = []

        contents = data[i]['context']
        contents = {c[0]: c[1] for c in contents}

        # collect the corpus
        if args.return_corpus:
            for title in contents:
                if title not in corpus_titles:
                    for i, context in enumerate(contents[title]):
                        corpus.append({'id': f"{len(corpus_titles)}:{i}", 'title': title, 'context': context})
                    corpus_titles.append(title)
                else:
                    continue

        for (title, sentid) in facts:
            if title in contents:
                try:
                    evidences.append({
                        "docid": f"{corpus_titles.index(title)}:{sentid}",
                        "title": title,
                        "context": contents[title][sentid]
                    })
                except:
                    print('incorrect document')
                    evidences.append({"title": title, "context": ""})

        fout.write(json.dumps({
            "qid": qid, "question": query,
            "answer": answer, "evidences": evidences
        }, ensure_ascii=False)+'\n')

    # given corpus a numbers
    for c in corpus:
        fcorpus.write(json.dumps(
            {'id': c['id'], 'title': c['title'], 'contents': c['context']},
            ensure_ascii=False
        )+'\n')
    print("Done")
