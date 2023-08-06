search_ikat_train:
	python3 search/bm25_ikat.py \
	    --k 100 --k1 0.9 --b 0.4 \
	    --index /home/jhju/indexes/full_wiki_segments_lucene_tc/ \
	    --output runs/ikat.train.bm25.run \
	    --resolved \
	    --topic data/ikat/2023_train_topics.json

search_ikat_test:
	python3 search/bm25_ikat.py \
	    --k 100 --k1 0.9 --b 0.4 \
	    --index /home/jhju/indexes/full_wiki_segments_lucene_tc/ \
	    --output runs/ikat.test.bm25.run \
	    --resolved \
	    --topic data/ikat/2023_test_topics.json

todo:
	echo 'hh'
