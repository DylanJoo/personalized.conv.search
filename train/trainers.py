import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
from transformers import (
    Trainer, 
    Seq2SeqTrainer,
    PreTrainedModel
)
from loss import InBatchNegativeCELoss as info_nce
from loss import PairwiseCELoss as pair_ce
from loss import LMCELoss as gen_ce

class TrainerForStart(Trainer):
    def __init__(
        self, 
        document_encoder: Union[PreTrainedModel, nn.Module] = None,
        temperature: Optional[float] = 1,
        **kwargs
    ):
        self.document_encoder = document_encoder
        self.temperature = temperature
        super().__init__(**kwargs)
        self.alpha = self.args.alpha

    def compute_loss(self, model, batch, return_outputs=False):
        # forward triplet and get logits
        outputs = model.forward_start(
                batch['q_inputs'], 
                batch['qs_inputs'],
                batch['p_inputs'],
                document_encoder=self.document_encoder
        )

        # Calculate losses
        ## 1) InfoNCE loss for dense retrieval
        qp_logits = outputs['qp_logits'] / self.temperature
        qsp_logits = outputs['qsp_logits'] / self.temperature
        loss_rt = info_nce(qp_logits) + info_nce(qsp_logits)

        ## 2) Relevance-aware pairwise loss for personalized retrieval
        paired_logits = torch.stack([
            torch.diag(qp_logits), torch.diag(qsp_logits)
        ], dim=-1)
        loss_start = pair_ce(paired_logits, pos_idx=1)

        ## 3) Schedule/weight loss
        loss = loss_rt + loss_start * self.alpha

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss

class TrainerForStarter(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model.forward(**inputs)

        # Calculate losses
        ## generation NLL/CE loss
        loss = outputs.get('loss', 0)
        # loss = gen_ce(
        #         outputs['logits'], inputs['labels'], 
        #         model.config.vocab_size
        # )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
