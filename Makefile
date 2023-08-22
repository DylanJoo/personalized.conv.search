# spare retrieval
CLUEWEB=/tmp2/trec/ikat/indexes/clueweb22_ikat-lucene
ssearch_ikat_train:
	python3 sparse_retrieval/bm25_ikat.py \
	    --k 100 --k1 0.9 --b 0.4 \
	    --index $CLUEWEB \
	    --output runs/ikat.train.bm25.run \
	    --resolved \
	    --topic data/ikat/2023_train_topics.json

ssearch_ikat_test:
	python3 sparse_retrieval/bm25_ikat.py \
	    --k 100 --k1 0.9 --b 0.4 \
	    --index $CLUEWEB \
	    --output runs/ikat.test.bm25.run \
	    --resolved \
	    --topic data/ikat/2023_test_topics.json

preprocess_hotpot_train:
	 python3 preprocess/flatten_hotpotqa.py \
	    --hotpotqa_json data/hotpotqa/hotpot_train_v1.1.json \
	    --output data/hotpotqa/flatten_hotpotqa.jsonl \
	    --return_corpus
	
index_hotpot_corpus_contriever:
	python dense_retrieval/index.py input \
	    --corpus data/hotpotqa/passages.jsonl \
	    --fields title contents \
	    --shard-id 0 \
	    --shard-num 1 output \
	    --embeddings /tmp2/trec/ikat/indexes/hotpotqa-contriever/ \
	    --to-faiss encoder \
	    --encoder-class contriever \
	    --encoder /tmp2/trec/pds/retrievers/contriever-msmarco/ \
	    --fields title contents \
	    --batch 32 \
	    --fp16 \
	    --device cuda:2

dsearch_hotpot_train:
	python dense_retrieval/search.py \
	    --k 200 \
	    --output runs/hotpotqa.train.contriever.run  \
	    --index /tmp2/trec/ikat/indexes/hotpotqa-contriever/ \
	    --encoder_path /tmp2/trec/pds/retrievers/contriever-msmarco/ \
	    --query data/hotpotqa/flatten_hotpotqa.jsonl \
	    --device cuda:1 \
	    --batch_size 64

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

dsearch_qrecc_train: 
	python3 dense_retrieval/search.py \
	    --k 10  \
	    --index /tmp2/trec/ikat/indexes/wikipedia-contriever/ \
	    --output runs/qrecc.train.contriever.rewrite.run \
	    --encoder_path /tmp2/trec/pds/retrievers/contriever-msmarco/ \
	    --query data/qrecc/qrecc_train.json \
	    --device cuda:2 \
	    --batch_size 64 

dsearch_ikat_train: 
	# use for response generation
	# use resolved question
	python3 dense_retrieval/search.py \
	    --k 100  \
	    --index /tmp2/trec/ikat/indexes/wiki-contriever/ \
	    --output runs/ikat.train.contriever.resolved.run \
	    --encoder_path /tmp2/trec/pds/retrievers/contriever-msmarco/ \
	    --query data/ikat/2023_train_topics.json \
	    --device cuda:2 \
	    --batch_size 64 \
	    --resolved
	
	# use ptkb 
	python3 dense_retrieval/search.py \
	    --k 100  \
	    --index /tmp2/trec/ikat/indexes/msmarco-contriever/ \
	    --output runs/ikat.train.contriever.resolved+ptkb.run \
	    --encoder_path /tmp2/trec/pds/retrievers/contriever-msmarco/ \
	    --query data/ikat/2023_train_topics.json \
	    --device cuda:2 \
	    --batch_size 64 \
	    --resolved \
	    --concat_ptkb

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

MODEL=DylanJHJ/gtr-t5-base
export CUDA_VISIBLE_DEVICES=2
train_start_gtr:
	python3 train/train_start_gtr.py \
     		--model_name_or_path ${MODEL} \
		--tokenizer_name ${MODEL} \
     		--train_file data/start/train.jsonl \
		--config_name ${MODEL} \
		--output_dir models/ckpt/start-base-B160 \
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
	        --alpha 0.1 

MODEL=facebook/contriever-msmarco
export CUDA_VISIBLE_DEVICES=2
ALPHA=0.5
train_start_contriever:
	python3 train/train_start_contriever.py \
     		--model_name_or_path ${MODEL} \
		--tokenizer_name ${MODEL} \
     		--train_file data/start/train.jsonl \
		--config_name ${MODEL} \
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
