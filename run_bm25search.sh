# run ikat 2023
# train
python3 search/bm25_ikat.py \
    --k 100 --k1 0.9 --b 0.4 \
    --index /home/jhju/indexes/full_wiki_segments_lucene_tc/ \
    --output runs/ikat.train.bm25.run \
    --resolved \
    --topic data/ikat/2023_train_topics.json

# test
python3 search/bm25_ikat.py \
    --k 100 --k1 0.9 --b 0.4 \
    --index /home/jhju/indexes/full_wiki_segments_lucene_tc/ \
    --output runs/ikat.test.bm25.run \
    --resolved \
    --topic data/ikat/2023_test_topics.json

# run cast 2021
# python3 search/bm25_cast.py \
#     --k 1000 --k1 0.68 --b 0.82 \
#     --index /home/jhju/indexes/full_wiki_segments_lucene/ \
#     --output runs/cast.eval.2021.bm25.run \
#     --topic data/cast/2021/2021_manual_evaluation_topics_v1.0.json
