import pickle
import time
import numpy as np
import os
from tqdm import tqdm 
import json
import argparse
from index import Indexer
from glob import glob

def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids

def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")

def main(embeddings_path, indexing_batch_size):
    index = Indexer(768, 0, 8)

    # index all passages
    input_paths = glob(embeddings_path)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index")

    print(f"Indexing passages from files {input_paths}")
    start_time_indexing = time.time()
    index_encoded_data(index, input_paths, indexing_batch_size)
    print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
    index.serialize(embeddings_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str)
    parser.add_argument("--indexing_batch_size", type=int, default=1000000)
    args = parser.parse_args()

    main(args.index_dir, args.indexing_batch_size)

