# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modifiy to start training

import os
import torch
from transformers import BertModel

class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward_start(self, 
                      q_inputs, qs_inputs, p_inputs, 
                      document_encoder=None, **kwargs):
        """ See the customized datacollator in `data.py` for detail.  """
        # 1) Get representation 
        ## document embeddings
        ### Setting 0: freezed document encoder
        if document_encoder is not None:
            D = document_encoder.encode(p_inputs, normalized=False)
        else:
            # [NOTE] To simplify the PoC, this is deprecated so far.
            d_output = self.forward(**p_inputs)
            D = self.mean_pooling(d_output, p_inputs['attention_mask'])

        ## query embeddings
        q_output = self.forward(**q_inputs)
        Q = self.mean_pooling(q_output, q_inputs['attention_mask'])

        ## statement-awared query embeddings
        qs_output = self.forward(**qs_inputs)
        QS = self.mean_pooling(qs_output, qs_inputs['attention_mask'])

        ## query-passage relevance logits
        qp_logits = Q @ D.transpose(0, 1)
        qsp_logits = QS @ D.transpose(0, 1)

        return {'qp_logits': qp_logits, 'qsp_logits': qsp_logits}

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, inputs, normalized=True):
        model_output = self.forward(**inputs)
        pooled_embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
        if normalized:
            encoded_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        else:
            encoded_embeddings = pooled_embeddings
        return encoded_embeddings
