import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
import json
import re


model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    pad_token_id=tokenizer.pad_token_id
)

# Themes dictionary
themes = {
    "Politics": "Write a sentence about politics: ",
    "Health": "Write a sentence about health and wellness: ",
    "Finance": "Write a sentence about finance or the economy: ",
    "Travel": "Write a sentence about travel or tourism: ",
    "Food": "Write a sentence about food and cooking: ",
    "Education": "Write a sentence about education: ",
    "Environment": "Write a sentence about environmental issues: ",
    "Fashion": "Write a sentence about fashion trends: ",
    "Science": "Write a sentence about science and discovery: ",
    "Sports": "Write a sentence about sports: ",
    "Technology": "Write a sentence about technology: ",
    "Entertainment": "Write a sentence about entertainment: "
}


def clean_text(text, prompt):
    cleaned_text = text.replace(prompt, "").strip()
    cleaned_text = re.sub(r'\u00a0', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('\n', '')
    cleaned_text = cleaned_text.replace('\\', '')
    cleaned_text = cleaned_text.replace('"', '')
    return cleaned_text
    
def generate_sentences(category_prompt, num_sentences=5):
    sentences = set()
    while len(sentences) < num_sentences:
        try:
            result = generator(
                category_prompt,
                max_length=30,
                num_return_sequences=1,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2
            )
            raw_text = result[0]['generated_text'].strip()
            cleaned_text = clean_text(raw_text, category_prompt)
            sentences.add(cleaned_text)
        except Exception as e:
            print(f"Error generating text: {e}")
            break
    return list(sentences)


expanded_data = {}
for category, prompt in themes.items():
    print(f"Generating sentences for category: {category}")
    expanded_data[category] = generate_sentences(prompt, num_sentences=100)


output_path = "cleaned_generated_train.json"
with open(output_path, 'w') as output_file:
    json.dump(expanded_data, output_file, indent=4)

print(f"Generated and cleaned data saved to {output_path}")
