import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_identifier = "snrspeaks/t5-one-line-summary"

tokenizer = AutoTokenizer.from_pretrained(model_identifier)
model = AutoModelForSeq2SeqLM.from_pretrained(model_identifier)

def get_summary(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    tok_len = len(input_ids[0])
    outputs = model.generate(input_ids, max_new_tokens=int(tok_len), min_length=int(tok_len/2))
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded