""" [TODO] retrofit the gtr encoders into instruction-tunned dense retriever
"""
import torch
import copy
from transformers import (
    T5EncoderModel, 
    T5Config
)
from transformers.models.t5.modeling_t5 import T5Stack
import torch.nn as nn
import torch.nn.functional as F

class GTREncoder(T5EncoderModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        self.linear = nn.Linear(config.d_model, config.d_model, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward_start(self, batch_inputs, document_encoder=None, **kwargs):
        """ See the customized datacollator in `data.py` for detail.
        """
        q_inputs, qs_inputs, p_inputs = batch_inputs 

        # 1) Get representation 
        ## document embeddings
        ### Setting 0: freezed document encoder
        if document_encoder is not None:
            D_embeds = document_encoder.encode(**p_inputs, normalized=False)
        else:
            # [NOTE] To simplify the PoC, this is deprecated so far.
            D_output = self.forward(**p_inputs)
            D_embeds = self.mean_pooling(D_output, p_inputs['attention_mask'])

        ## query embeddings
        Q_output = self.forward(**q_inputs)
        Q_embeds = self.mean_pooling(Q_output, q_inputs['attention_mask'])

        ## statement-awared query embeddings
        QS = self.forward(**qs_inputs)
        QS_embeds = self.mean_pooling(QS_output, qs_inputs['attention_mask'])

        # 2) InfoNCE loss
        logits_q = Q_embeds @ D_embeds.transpoe()
        logits_qs = QS_embeds @ D_embeds.transpoe()
        labels = torch.eye(logits_q.size(0), device=self.device)
        loss_qqs = loss(logits_q, labels) + loss(logits_qs, labels)

        # 3) Relevance-aware contrastive learning
        logits_diag = torch.cat(
                [torch.diag(logits_q), torch.diag(logits_qs)], dim=1
        )
        labels = torch.tensor([[0, 1]]*logits_q.size(0), device=self.device)
        loss_diag = loss(logits_diag, labels)

        loss = loss_qqs + loss_diag
        return {'loss': loss, 'score': scores}

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, **inputs, normalized=True):
        model_output = self.forward(**inputs)
        pooled_embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
        if normalized:
            encoded_embeddings = F.normalize(self.linear(pooled_embeddings), p=2, dim=1)
        else:
            encoded_embeddings = self.linear(pooled_embeddings)
        return encoded_embeddings
