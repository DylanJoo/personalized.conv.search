import torch
from typing import Optional, Union
import torch.nn as nn
from transformers import Trainer, PreTrainedModel
from loss import InBatchNegativeCELoss as info_nce
from loss import PairwiseCELoss as pair_ce

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

    def compute_loss(self, model, inputs, return_outputs=False):
        # the label is constructed by InfoNCE, so no `label`
        q_inputs, qs_inputs, p_inputs = inputs

        # forward triplet and get logits
        outputs = model.forward_start(
                q_inputs, qs_inputs, p_inputs, 
                document_encoder=self.document_encoder
        )

        # Calculate losses
        ## 1) InfoNCE loss for dense retrieval
        qp_logits = outputs['qp_logits'] / self.temperature
        qsp_logits = outputs['qsp_logits'] / self.temperature
        ### testing
        m = nn.Softmax(dim=-1)
        print(m(qp_logits)[0:3, 0:3])
        print(m(qsp_logits)[0:3, 0:3])

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
