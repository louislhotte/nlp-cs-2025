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
    "Politics": "Write a sentence about a current political issue, debate, or development in 20 words or less.",
    "Health": "Write a sentence about a topic related to health, wellness, or medical advancements in 20 words or less.",
    "Finance": "Write a sentence about personal finance, the economy, or financial markets in 20 words or less.",
    "Travel": "Write a sentence about a travel destination, tourism, or an adventure experience in 20 words or less.",
    "Food": "Write a sentence about a culinary dish, recipe, or food culture in 20 words or less.",
    "Education": "Write a sentence about an educational trend, policy, or innovation in learning in 20 words or less.",
    "Environment": "Write a sentence about an environmental concern, solution, or sustainability effort in 20 words or less.",
    "Fashion": "Write a sentence about a popular trend, style, or shift in the fashion industry in 20 words or less.",
    "Science": "Write a sentence about a scientific breakthrough, research, or discovery in 20 words or less.",
    "Sports": "Write a sentence about an event, player, or trend in the world of sports in 20 words or less.",
    "Technology": "Write a sentence about a technological innovation, gadget, or its impact on society in 20 words or less.",
    "Entertainment": "Write a sentence about a film, music, or cultural phenomenon in entertainment in 20 words or less."
}



def clean_text(text, prompt):
    cleaned_text = text.replace(prompt, "").strip()
    cleaned_text = re.sub(r'\u00a0', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('\n', '')
    cleaned_text = cleaned_text.replace('\\', '')
    cleaned_text = cleaned_text.replace('"', '')
    cleaned_text = cleaned_text.replace('\u2014', '')
    cleaned_text = cleaned_text.replace('\u00b4', '')
    cleaned_text = cleaned_text.replace('_', '')
    cleaned_text = cleaned_text.replace('\u00a9', '')
    cleaned_text = re.sub(r'\\u\w+', '', cleaned_text)
    return cleaned_text
    
def generate_sentences(category_prompt, num_sentences=5):
    sentences = set()
    while len(sentences) < num_sentences:
        try:
            result = generator(
                category_prompt,
                max_length=60,
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
    expanded_data[category] = generate_sentences(prompt, num_sentences=200)


output_path = "cleaned_generated_train.json"
with open(output_path, 'w') as output_file:
    json.dump(expanded_data, output_file, indent=4)

print(f"Generated and cleaned data saved to {output_path}")