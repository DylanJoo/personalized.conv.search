import torch
# from models import FiDT5_meta
from transformers import T5ForConditionalGeneration

# load fid model checkppints
model_old = torch.load('models/ckpt/fidt5-base-nq/pytorch_model.bin', map_location='cpu')

model_new = T5ForConditionalGeneration.from_pretrained('t5-base')

# compare state dict
model_new_keys = sorted(list(model_new.state_dict().keys()))
model_old_keys = sorted(list(model_old.keys()))

# change key map
for k in model_old_keys:
    k_prime = k.replace('encoder.encoder', 'encoder')
    k_prime = k_prime.replace('module.layer', 'layer')
    model_old[k_prime] = model_old.pop(k)

# validate if the old keys align the new one
# model_old_keys = sorted(list(model_old.keys()))
#
# for i, k in enumerate(model_new_keys):
#     if k not in model_old_keys:
#         print(model_old_keys[i])
#         print(k)

# save as the new checkpoint
torch.save(model_old, '/home/jhju/models/fidt5-base-nq/pytorch_model.bin')
