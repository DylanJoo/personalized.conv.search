# python -m pyserini.index.lucene \
#   --collection JsonCollection \
#   --input $COLLECTION_DIR \
#   --index $INDEX \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 4 \
#   --storePositions --storeDocvectors --storeRaw
