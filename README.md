# TREC iKAT: personalized.conv.search
Personalized conversational search for trec ikat

## ConvSearch
- Baseline ad-hoc search
the following scripts are the `baseline` 100 retrieved passages from WIKI corpus. The topic files are `2023_train_topics.json` and `2023_test_topics.json`
. 
Performing BM25 search with `raw utterance` or `resolved utterance`.
```
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
```


## Data
- Collection
    * WIKI full
    * ClueWeb
- Text retrieval dataset
    * WIKI full (Lucene): '/home/jhju/indexes/full_wiki_segments_lucene'
    * WIKI full (DPR-multi): '/home/jhju/indexes/full_wiki_segments_dpr'
- PTKB ranking
    * Synthesized

### Methods
- ConvSearch
    * CQE+ConvRerank
