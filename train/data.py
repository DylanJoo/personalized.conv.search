""" The datacollator for pcentric dataset.
"""
import torch
import random
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

@dataclass
class DataCollatorForRACtxGenerator:
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
                padding=self.padding,
                return_tensors='pt'
        )
        qs_inputs = self.tokenizer(
                [f"{q} </s> {s}" for q, s in zip(queries, statements)],
                max_length=self.max_q_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        )

        # passage 
        ## in-batch negative [TODO] hard negative
        p_inputs = self.tokenizer(
                passages,
                max_length=self.max_p_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        )
        
        return {'q_inputs': q_inputs, 
                'qs_inputs': qs_inputs,
                'p_inputs': p_inputs, 
                'return_loss': True}

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
                padding=self.padding,
                return_tensors='pt'
        )
        qs_inputs = self.tokenizer(
                [f"{q} </s> {s}" for q, s in zip(queries, statements)],
                max_length=self.max_q_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        )

        # passage 
        ## in-batch negative [TODO] hard negative
        p_inputs = self.tokenizer(
                passages,
                max_length=self.max_p_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        )
        
        return {'q_inputs': q_inputs, 
                'qs_inputs': qs_inputs,
                'p_inputs': p_inputs, 
                'return_loss': True}
