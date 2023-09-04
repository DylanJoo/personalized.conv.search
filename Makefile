# spare retrieval
CLUEWEB=/tmp2/trec/ikat/indexes/clueweb_ikat/
ssearch_ikat_train:
	python3 sparse_retrieval/bm25_ikat.py \
	    --k 1000 --k1 0.9 --b 0.4 \
	    --index ${CLUEWEB} \
	    --output runs/ikat.train.bm25.run \
	    --rewritten data/ikat/kat.train.t5ntr_history_3-3.jsonl \
	    --topic data/ikat/2023_train_topics.json

ssearch_ikat_test:
	python3 sparse_retrieval/bm25_ikat.py \
	    --k 1000 --k1 0.9 --b 0.4 \
	    --index ${CLUEWEB} \
	    --output runs/ikat.test.bm25.run \
	    --rewritten data/ikat/kat.test.t5ntr_history_3-3.jsonl \
	    --topic data/ikat/2023_test_topics.json

index_ikat_clueweb:
	python3 -m pyserini.index.lucene \
	    --collection JsonCollection \
	    --input /tmp2/trec/ikat/data/collection/ikat/ \
	    --index /tmp2/trec/ikat/indexes/clueweb_ikat/ \
	    --generator DefaultLuceneDocumentGenerator \
	    --threads 4

dsearch_qrecc_train: 
	python3 dense_retrieval/search.py \
	    --k 10  \
	    --index /home/jhju/indexes/wikipedia-contriever/ \
	    --output runs/qrecc.train.contriever.rewrite.run \
	    --encoder_path /tmp2/trec/pds/retrievers/contriever-msmarco/ \
	    --query data/qrecc/qrecc_train.json \
	    --device cuda:2 \
	    --batch_size 64 

dsearch_ikat_train_baseline: 
	# baseline (msmarco contriever)
	python3 dense_retrieval/search.py \
	    --k 20  \
	    --index /tmp2/trec/ikat/indexes/wiki-contriever/ \
	    --index /home/jhju/indexes/wikipedia-contriever/ \
	    --output runs/ikat.train.contriever.wiki.run \
	    --encoder_path /tmp2/trec/pds/retrievers/contriever-msmarco/ \
	    --query data/ikat/2023_train_topics.json \
	    --rewritten data/ikat/ikat.train.t5ntr_history_3-3.jsonl \
	    --device cuda:2 \
	    --batch_size 8

dsearch_ikat_train:
	# start-contriver # use ptkb
	python3 dense_retrieval/search.py \
	    --k 10  \
	    --index /home/jhju/indexes/wikipedia-contriever/ \
	    --output runs/ikat.train.start-contriever.wiki.run \
	    --encoder_path models/ckpt/start-contriever-ms-B160-A0.1/checkpoint-20000/ \
	    --query data/ikat/2023_train_topics.json \
	    --rewritten data/ikat/ikat.train.t5ntr_history_3-3.jsonl \
	    --device cuda:2 \
	    --concat_ptkb \
	    --batch_size 8
	# [NOTE] ikat-subtask3 required.

dsearch_ikat_test_baseline:
	# baseline (msmarco contriever)
	python3 dense_retrieval/search.py \
	    --k 10  \
	    --index /home/jhju/indexes/wikipedia-contriever/ \
	    --output runs/ikat.test.contriever.wiki.run \
	    --encoder_path /tmp2/trec/pds/retrievers/contriever-msmarco/ \
	    --query data/ikat/2023_test_topics.json \
	    --rewritten data/ikat/ikat.test.t5ntr_history_3-3.jsonl \
	    --device cuda:2 \
	    --batch_size 16

dsearch_ikat_test:
	# start-contriver # use ptkb 
	python3 dense_retrieval/search.py \
	    --k 10  \
	    --index /home/jhju/indexes/wikipedia-contriever/ \
	    --output runs/ikat.test.start-contriever.wiki.run \
	    --encoder_path models/ckpt/start-contriever-ms-B160-A0.1/checkpoint-20000/ \
	    --query data/ikat/2023_test_topics.json \
	    --rewritten data/ikat/ikat.test.t5ntr_history_3-3.jsonl \
	    --device cuda:2 \
	    --concat_ptkb \
	    --batch_size 16
	# [NOTE] ikat-subtask3 required.

construct_start_train:
	python3 augmentation/convert_llm_to_start_train.py.py \
    		--collection /tmp2/trec/ikat/data/collection/wiki/wiki_psgs_w100.tsv  \
    		--input_jsonl /home/jhju/huggingface_hub/START/train.unprocessed.jsonl \
     		--output_jsonl data/start/train.jsonl 
	# Consistenct filter
	# python3 augmentation/convert_llm_to_start_train.py.py \
    	# 	--input_jsonl \
     	# 	--output_jsonl \
     	# 	--filtering \
     	# 	--filter_model str \
     	# 	--filter_k int \
     	# 	--filter_thres float 

construct_starter_train:
	python3 augmentation/convert_llm_to_starter_train.py \
    		--collection /tmp2/trec/ikat/data/collection/wiki/wiki_psgs_w100.tsv  \
    		--run runs/qrecc.train.contriever.rewrite.run \
    		--topic data/qrecc/qrecc_train.json \
    		--input_jsonl /home/jhju/huggingface_hub/START/train.unprocessed.jsonl \
     		--output_jsonl data/start/train_starter.jsonl 
ALPHA=0.5
train_start_gtr:
	export CUDA_VISIBLE_DEVICES=2; python3 train/train_start_gtr.py \
     		--model_name_or_path DylanJHJ/gtr-t5-base \
		--tokenizer_name DylanJHJ/gtr-t5-base \
     		--train_file data/start/train.jsonl \
		--config_name DylanJHJ/gtr-t5-base \
		--output_dir models/ckpt/start-gtr-base-B160-A${ALPHA} \
	        --max_p_length 256 \
	        --max_q_length 64 \
	        --per_device_train_batch_size 160 \
	        --learning_rate 1e-5 \
	        --evaluation_strategy steps \
	        --max_steps 20000 \
	        --save_steps 5000 \
	        --eval_steps 500 \
	        --freeze_document_encoder true \
	        --do_train \
	        --do_eval \
	        --optim adafactor \
	        --temperature 0.25 \
	        --alpha ${ALPHA}

ALPHA=0.5
train_start_contriever:
	export CUDA_VISIBLE_DEVICES=2
	python3 train/train_start_contriever.py \
     		--model_name_or_path facebook/contriever-msmarco \
		--tokenizer_name facebook/contriever-msmarco \
     		--train_file data/start/train.jsonl \
		--config_name facebook/contriever-msmarco \
		--output_dir models/ckpt/start-contriever-ms-B160-A0.5 \
	        --max_p_length 256 \
	        --max_q_length 64 \
	        --per_device_train_batch_size 160 \
	        --learning_rate 1e-5 \
	        --evaluation_strategy steps \
	        --max_steps 20000 \
	        --save_steps 5000 \
	        --eval_steps 500 \
	        --freeze_document_encoder true \
	        --do_train \
	        --do_eval \
	        --optim adamw_hf \
	        --warmup_steps 800 \
	        --temperature 0.25 \
	        --alpha ${ALPHA}

precompute_star_embeds:
	python3 train/generate_star_embeds.py \
     		--model_name_or_path DylanJHJ/gtr-t5-base \
		--input_jsonl data/start/train_starter.jsonl \
		--output_jsonl data/start/temp.jsonl \
		--max_num_statements 10 \
		--min_num_statements 5 \
	        --max_length 64 \
		--device 'cuda:2' \
		--sep_token '</s>'

train_starter_fid:
	export CUDA_VISIBLE_DEVICES=2; python3 train/train_starter_fid.py \
     		--model_name_or_path DylanJHJ/fidt5-base-nq \
     		--tokenizer_name DylanJHJ/fidt5-base-nq \
		--config_name DylanJHJ/fidt5-base-nq \
     		--train_file data/start/train_starter_embeds.jsonl \
		--output_dir models/ckpt/starter-fid-gtr-B32 \
	        --per_device_train_batch_size 32 \
	        --max_src_length 320 \
	        --max_tgt_length 32 \
	        --precomputed_star_embeds true \
	        --retrieval_enhanced true \
	        --learning_rate 1e-5 \
	        --evaluation_strategy steps \
	        --max_steps 20000 \
	        --save_steps 5000 \
	        --eval_steps 500 \
	        --do_train \
	        --do_eval \
	        --optim adafactor \
	        --warmup_steps 800


# preprocess_hotpot_train:
# 	python3 preprocess/flatten_hotpotqa.py \
# 	    --hotpotqa_json data/hotpotqa/hotpot_train_v1.1.json \
# 	    --output data/hotpotqa/flatten_hotpotqa.jsonl \
# 	    --return_corpus

# index_hotpot_corpus_contriever:
# 	python dense_retrieval/index.py input \
# 	    --corpus data/hotpotqa/passages.jsonl \
# 	    --fields title contents \
# 	    --shard-id 0 \
# 	    --shard-num 1 output \
# 	    --embeddings /tmp2/trec/ikat/indexes/hotpotqa-contriever/ \
# 	    --to-faiss encoder \
# 	    --encoder-class contriever \
# 	    --encoder /tmp2/trec/pds/retrievers/contriever-msmarco/ \
# 	    --fields title contents \
# 	    --batch 32 \
# 	    --fp16 \
# 	    --device cuda:2
#
# dsearch_hotpot_train:
# 	python dense_retrieval/search.py \
# 	    --k 200 \
# 	    --output runs/hotpotqa.train.contriever.run  \
# 	    --index /tmp2/trec/ikat/indexes/hotpotqa-contriever/ \
# 	    --encoder_path /tmp2/trec/pds/retrievers/contriever-msmarco/ \
# 	    --query data/hotpotqa/flatten_hotpotqa.jsonl \
# 	    --device cuda:1 \
# 	    --batch_size 64

# evaluate_hotpot:
# 	/tmp2/trec/trec_eval.9.0.4/trec_eval \
# 	    -c -m recall.10 data/hotpotqa/od_hotpot_train_avail.qrel \
# 	    runs/hotpotqa.train.contriever.run

# collect_triplet:
# 	python augmentation/collect_triplet.py \
# 	    --flatten_qa data/hotpotqa/flatten_hotpotqa.jsonl \
# 	    --run runs/hotpotqa.train.contriever.run \
# 	    --corpus data/hotpotqa/passaegs.jsonl \
# 	    --triplet data/ikat/hotpotqa_triplet.jsonl

# predict_questions:
# 	python augmentation/generate_question.py \
# 	    --corpus data/hotpotqa/passages.jsonl \
# 	    --triplet data/ikat/hotpotqa_triplet.jsonl \
# 	    --flatten_qa data/hotpotqa/flatten_hotpotqa.jsonl \
# 	    --prediction data/ikat/hotpotqa_prediction.jsonl  \
# 	    --model_name mrm8488/t5-base-finetuned-question-generation-ap \
# 	    --device cuda:2

rerank_contriever_baseline:
	python3 dense_retrieval/rerank_contriever.py \
	    --q_encoder_path models/ckpt/start-contriever-ms-B160-A0.1/checkpoint-20000/ \
	    --d_encoder_path /home/jhju/models/contriever-msmarco/ \
	    --device cuda:2 \
	    --batch_size 8 \
	    --run runs/ikat.test.bm25.run \
	    --query data/ikat/2023_test_topics.json \
	    --rewritten data/ikat/ikat.test.t5ntr_history_3-3.jsonl \
	    --collection_dir /tmp2/trec/ikat/data/collection/ikat/ \
	    --output_run runs/ikat.test.bm25.contriever.run
