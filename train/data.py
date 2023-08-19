""" The datacollator for pcentric dataset.
"""
import torch
import random
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

@dataclass
class DataCollatorForCtxRetriever:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_p_length: Optional[int] = 128
    max_q_length: Optional[int] = 64

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # raw query
        queries, statements, passages = [], [], []
        for batch in features:
            queries.append(batch['query'])
            statements.append(batch['statement'])
            passages.append(batch['passage'])


        # query and statement-awared query
        q_inputs = self.tokenizer(
                queries,
                max_length=self.max_q_length,
                truncation=self.truncation,
                padding=self.truncation,
                return_tensors='pt'
        )
        qs_inputs = self.tokenizer(
                [f"{q} </s> {s}" for q, s in zip(queries, statements)],
                max_length=self.max_q_length,
                truncation=self.truncation,
                padding=self.truncation,
                return_tensors='pt'
        )

        # passage 
        ## in-batch negative [TODO] hard negative
        p_inputs = self.tokenizer(
                passages,
                max_length=self.max_p_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
        )
        
        return q_inputs, qs_inputs, p_inputs

# class QReCC:
#
#     def __init__(self, path_query, path_collection):
#         self.query = load_query(path_query)
#         self.document = load_collection(path_collection)
#         self.dataset = None
#
#     def create_dataset_from_run(self, path):
#         data_list = []
#
#         with open(path, 'r') as fin:
#             for line in tqdm(fin):
#                 qid, _, docid, rank, score, _ = line.split()
#                 data_list.append({
#                     "query": self.query[qid],
#                     "statement": self.query[qid],
#                     "passage": self.query[qid],
#                 })
#
#             dataset = Dataset.from_list()


