python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/jhju/datasets/wiki.full_wiki_segments/tc/ \
  --index /home/jhju/indexes/full_wiki_segments_lucene_tc \
  --generator DefaultLuceneDocumentGenerator \
  --threads 32
