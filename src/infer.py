import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_identifier = "snrspeaks/t5-one-line-summary"

tokenizer = AutoTokenizer.from_pretrained(model_identifier)
model = AutoModelForSeq2SeqLM.from_pretrained(model_identifier)

def get_summary(input_text):
    input_ids = tokenizer.encode(
        "summarize: " + input_text, return_tensors="pt", add_special_tokens=True
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        num_beams=5,
        max_length=512,
        repetition_penalty=2.5,
        length_penalty=1,
        early_stopping=True,
        num_return_sequences=1,
    )

    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return summary