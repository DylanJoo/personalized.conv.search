search_ikat_train:
	python3 sparse_retrieval/bm25_ikat.py \
	    --k 100 --k1 0.9 --b 0.4 \
	    --index /home/jhju/indexes/full_wiki_segments_lucene_tc/ \
	    --output runs/ikat.train.bm25.run \
	    --resolved \
	    --topic data/ikat/2023_train_topics.json

search_ikat_test:
	python3 sparse_retrieval/bm25_ikat.py \
	    --k 100 --k1 0.9 --b 0.4 \
	    --index /home/jhju/indexes/full_wiki_segments_lucene_tc/ \
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

search_hotpot_contriever:
	python dense_retrieval/search.py \
	    --k 200 \
	    --output runs/hotpotqa.train.contriever.run  \
	    --index /tmp2/trec/ikat/indexes/hotpotqa-contriever/ \
	    --encoder_path /tmp2/trec/pds/retrievers/contriever-msmarco/ \
	    --query data/hotpotqa/flatten_hotpotqa.jsonl \
	    --device cuda:1 \
	    --batch_size 64
