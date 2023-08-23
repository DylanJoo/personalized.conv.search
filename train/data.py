""" The datacollator for pcentric dataset.
"""
import torch
import random
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers import T5EncoderModel
from transformers.tokenization_utils_base import (
    PaddingStrategy, 
    PreTrainedTokenizerBase
)

@dataclass
class DataCollatorForStarter:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_src_length: Optional[int] = 288
    max_tgt_length: Optional[int] = 32
    sep_token: Optional[str] = '</s>'
    n_contexts: Optional[int] = 1
    question_prefix: Optional[str] = 'question:'
    title_prefix: Optional[str] = 'title:'
    passage_prefix: Optional[str] = 'passage:'
    retrieval_enhanced: Union[bool, str] = False
    star_encoder: Optional[T5EncoderModel] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # [TODO] Sometimes title are lack, fix the template if needed.
        # [TODO] Empty-question assertion cases 

        # prepare input/output
        sources, targets = [], []
        for batch in features:

            ## context with question 
            ### multiple passages in a list for one question
            avail_contexts = batch['contexts'] + [("", "")]*self.n_contexts
            for i, context in enumerate(avail_contexts[:self.n_contexts]):
                sources.append( (batch['question'], context[0], context[1]) )

            ## answer
            ### [NOTE] Try different chat-ish target (e.g., DialoGPT)
            targets.append(batch['answer'])

        # preprocess src/tgt
        ## input_ids: (BN, L)
        ## attention_mask: (B, NL)
        ## labels: (B, L_tgt)
        template  = self.question_prefix + " {} "
        template += self.title_prefix + " {} " 
        template += self.passage_prefix + " {}"

        inputs = self.tokenizer(
                [template.format(q, t, p) for (q,t,p) in sources],
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        )
        inputs['labels'] = self.tokenizer(
                targets,
                max_length=self.max_tgt_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
        ).input_ids

        inputs['input_ids'] = inputs['input_ids'].view(
                -1, self.n_contexts, inputs['input_ids'].size(-1)
        )
        inputs['attention_mask'] = inputs['attention_mask'].view(
                -1, self.n_contexts*inputs['attention_mask'].size(-1)
        )

        ## past_key_values: (B, M); M stands # of statements(-aware) query
        if self.retrieval_enhanced:
            ### To use different amount of statements within a same batch, 
            ### fix the prefix embeddings into the same batch
            ### the size should be: (B N_statements H)
            inputs['past_key_values'] = torch.stack([
                torch.tensor(batch['statement_aware_embeds']) \
                        for batch in features
            ], dim=0)
        else:
            inputs['past_key_values'] = None
        
        return inputs

@dataclass
class DataCollatorForStart:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_p_length: Optional[int] = 128
    max_q_length: Optional[int] = 64
    sep_token: Optional[str] = '</s>'

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
                [f"{q} {self.sep_token} {s}" for q, s in zip(queries, statements)],
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
