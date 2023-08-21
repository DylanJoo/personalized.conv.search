from fidt5 import FiDT5
from transformers import AutoTokenizer

model = FiDT5.from_pretrained('DylanJHJ/fidt5-base-nq')
tokenizer = AutoTokenizer.from_pretrained('DylanJHJ/fidt5-base-nq')

inputs = tokenizer(
        ['apple 123', 'apple 456', 'apple 789', 'banana 123', 'banana 456', 'banana 789'], return_tensors='pt'
)

input_ids = inputs['input_ids'].view(2, 3, -1)
attention_mask = inputs['attention_mask'].view(2, -1)


output = model.generate(input_ids=input_ids, attention_mask=attention_mask)
print(output)
print(tokenizer.decode(output[0]))
