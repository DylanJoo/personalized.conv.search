from FiD.models import FiDT5
from transformers import T5Tokenizer, T5ForConditionalGeneration

# t5 = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
# tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = FiDT5(t5.config)
model.load_t5(t5.state_dict())

texts = ["Today is a sunday", "Today is a ranining day", "Today is a holiday."]
texts = [f"question: What is today's weather and is it holiday? passage: {t}" for t in texts]
tokenized_inputs = tokenizer(texts, return_tensors='pt', padding=True)
inputs = tokenized_inputs['input_ids'].unsqueeze(0)
attention_mask = tokenized_inputs['attention_mask'].unsqueeze(0)
outputs = model.generate(input_ids=inputs, attention_mask=attention_mask, max_length=50)
print(tokenizer.decode(outputs[0]))
