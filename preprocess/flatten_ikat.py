import os
from tqdm import tqdm 
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ikat_json", type=str)
    parser.add_argument("--output", default='flatten_ikat.jsonl', type=str)
    parser.add_argument("--return_corpus", action='store_true', default=False)
    args = parser.parse_args()

    # load data
    data = json.load(open(args.ikat_json, 'r')) 

    # writer
    fout = open(args.output, 'w')

    query = {}
    data = json.load(open(args.ikat_json, 'r'))
    for topic in data:
        topic_id = topic['number']
        try:
            turns = topic['turns']
            title = topic['title'] # more like a topic description
            ptkbs = topic['ptkb']
        except:
            # this is not the first turn of the topic
            continue

        for turn in turns:
            turn_id = turn['turn_id']
            question = turn['utterance'].strip()

            if args.resolved:    # [Prep 1]: resolved utterances
                question = turn['resolved_utterance'].strip()
            if args.concat_ptkb: # [Prep 2]: concat ptkb
                query[f"{topic_id}_{turn_id}"] = f"{ptkb} {question}"
            else:
                query[f"{topic_id}_{turn_id}"] = question

            response = turn['response']
            # ptkb_pvn = turn['ptkb_provenance']
            # response_pvn = turn['response_provenance']

    # 
    for topic in data:
        topic_id = topic['number']
        try:
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
