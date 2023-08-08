import argparse
import os
from tqdm import tqdm 
import json
import argparse
import sys
from tool import load_collection, load_topics
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_question(model, tokenizer, answers, docs, device='cpu'):
    processed_input = tokenizer(
            [f"answer: {a} context: {c}" for (a, c) in zip(answers, docs)],
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors='pt'
    ).to(device)

    # generate
    outputs = model.generate(**processed_input)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default='passages.jsonl', type=str)
    parser.add_argument("--triplet", default='hotpotqa_for_state_cont.jsonl', type=str)
    parser.add_argument("--flatten_qa", default='flatten_hotpotqa.jsonl', type=str)
    parser.add_argument("--prediction", default='prediction.jsonl', type=str)
    # models
    parser.add_argument("--model_name", default='mrm8488/t5-base-finetuned-question-generation-ap', type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    # writer
    fout = open(args.prediction, 'w')

    # load data
    collection = load_collection(args.corpus)
    questions = load_topics(args.flatten_qa, selected_key='question')
    answers = load_topics(args.flatten_qa, selected_key='answer')

    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    with open(args.triplet, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            qid = item['qid']
            question = questions[qid]
            answer = answers[qid]

            # candidate docs
            positive_docids = item['positive_docids']
            negative_docids = item['negative_docids']

            # construct the pair
            predicted_questions = generate_question(
                    model, tokenizer, answer, 
                    positive_docids+negative_docids,
                    device=args.device
            )

            # hard negative from contriever
            n_pos = len(positive_docids)
            for pred in predicted_questions:
                fout.write(json.dumps({
                    'qid': qid, 
                    'question_base': question, 
                    'positive_docids': positive_docids,
                    'positive_context': pred[:n_pos], 
                    'negative_docids': negative_docids,
                    'negative_context': pred[n_pos:]
                })+'\n')
