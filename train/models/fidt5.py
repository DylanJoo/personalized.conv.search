import copy
import torch
from transformers import T5ForConditionalGeneration, T5Config
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import  Seq2SeqLMOutput, BaseModelOutput
from transformers.models.t5.modeling_t5 import T5Stack

class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FiDT5Stack(encoder_config, self.shared) # replace 

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # add an adapter layer for adopting GTR embeddings
        ## [NOTE] the weights are random initialized
        self.embed_size_per_head = config.d_model // config.num_heads
        # Linear [H n_layer*H]
        self.proj_star = nn.Linear(
                config.d_model,
                config.num_decoder_layers * config.d_model,
                bias=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_crossattention_scores(self, context_mask):
        raise NotImplementedError('Please implement this function.')

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        if past_key_values is not None:
            if past_key_values.dim() == 3:
                # the first token generation (from star embeddings)
                # [NOTE] Add docstrings
                B, M = past_key_values.shape[:2]
                projection_star = self.proj_star(past_key_values)
                # B M H --> B L 12*(12*64)

                # B M L(12) nH(12) hH(64)
                # L(12) B nH(12) M hH(64) --> 2 0 3 1 4
                layer_star_embeds = projection_star.view(
                        B, 
                        M,
                        self.config.num_decoder_layers,
                        self.config.num_heads, 
                        self.embed_size_per_head
                ).permute(2, 0, 3, 1, 4)

                past_key_values = tuple(
                        (sa, sa, None, None) for sa in layer_star_embeds
                )


        # original forward passing 
        ## Encoder would not use it, only for decoding.
        return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values, # we dont need injection here
                encoder_outputs=encoder_outputs, 
                **kwargs
        )

    # def forward():
    # the `past_key_values` has been disabled in encoder forward;
    # however, it will activated as (additional) key/value states
    # in the decoder's self attention layer.

class FiDT5Stack(T5Stack):

    def forward(self, input_ids, attention_mask, **kwargs):
        """ Wrap/unwrap input/ouput with this class (replace t5-encoder) 

        :param input_ids: the tokenized input ids with shape (BN, L)
        :param attention_mask: the attention mask with shape (B, NL)

        :return encoder_outputs: the huggingface model output class.
        """
        if input_ids.dim() == 3: # normal usage of FiD
            B, N, L = input_ids.size()
        else:
            B, L = input_ids.size()
            N = 1

        # Modifying 1
        ## For `input_ids`, 
        ## transform from original batch into enuemrated batch.
        ## i.e. from (B, N, L) to (BN, L) 
        input_ids = input_ids.view(B*N, -1)
        ## For `attention_mask`, 
        ## transform from original batch into enuemrated batch.
        ## i.e. from (B, NL) to (BN, L) 
        attention_mask = attention_mask.view(B*N, -1)

        # Minor modifying
        ## Prefix tuning is used at decoder. 
        ## Avoid `past_key_value` being considered here (encoder)
        encoder_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                past_key_values=None, # preserve this for clarification.
                **kwargs
        )

        # Modifying 2
        ## transform from enuemrated batch into original batch 
        ## I.e. from (BN, L, H) to (B, NL, H) 
        encoder_outputs['last_hidden_state'] = \
                encoder_outputs['last_hidden_state'].view(B, N*L, -1)

        return encoder_outputs

