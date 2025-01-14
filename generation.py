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
    "Politics": "Example 1: \"The mayor announced a new initiative to improve public transportation.\"\nExample 2: \"The senator is facing criticism for her stance on the recent bill.\"\nExample 3: \"The upcoming election has sparked intense debates among the candidates.\"\nTask: Write a similar sentence about a political issue.",
    "Health": "Example 1: \"Regular exercise and a balanced diet are key to maintaining good health.\"\nExample 2: \"The World Health Organization has issued new guidelines on COVID-19.\"\nExample 3: \"A new study reveals the benefits of meditation for mental health.\"\nTask: Write a similar sentence about health.",
    "Finance": "Example 1: \"The stock market saw a significant drop following the announcement.\"\nExample 2: \"Investing in real estate can be a profitable venture if done correctly.\"\nExample 3: \"The company's profits have doubled since the launch of their new product.\"\nTask: Write a similar sentence about finance.",
    "Travel": "Example 1: \"Visiting the Grand Canyon is a breathtaking experience.\"\nExample 2: \"The tourism industry has been severely impacted by the pandemic.\"\nExample 3: \"Backpacking through Europe is a popular choice for young travelers.\"\nTask: Write a similar sentence about travel.",
    "Food": "Example 1: \"The new restaurant in town offers a fusion of Italian and Japanese cuisine.\"\nExample 2: \"Drinking eight glasses of water a day is essential for staying hydrated.\"\nExample 3: \"Cooking classes are a fun way to learn new recipes and techniques.\"\nTask: Write a similar sentence about food.",
    "Education": "Example 1: \"The school district is implementing a new curriculum for the upcoming year.\"\nExample 2: \"Online learning has become increasingly popular during the pandemic.\"\nExample 3: \"The university is offering scholarships for students in financial need.\"\nTask: Write a similar sentence about education.",
    "Environment": "Example 1: \"Climate change is causing a significant rise in sea levels.\"\nExample 2: \"Recycling and composting are effective ways to reduce waste.\"\nExample 3: \"The Amazon rainforest is home to millions of unique species.\"\nTask: Write a similar sentence about the environment.",
    "Fashion": "Example 1: \"The new fashion trend is all about sustainability and eco-friendly materials.\"\nExample 2: \"The annual Met Gala is a major event in the fashion world.\"\nExample 3: \"Vintage clothing has made a comeback in recent years.\"\nTask: Write a similar sentence about fashion.",
    "Science": "Example 1: \"NASA's Mars Rover has made significant discoveries about the red planet.\"\nExample 2: \"The Nobel Prize in Physics was awarded for breakthroughs in black hole research.\"\nExample 3: \"Genetic engineering is opening up new possibilities in medical treatment.\"\nTask: Write a similar sentence about science.",
    "Sports": "Example 1: \"The NBA Finals are set to begin next week with the top two teams in the league.\"\nExample 2: \"Serena Williams continues to dominate the tennis world with her powerful serve.\"\nExample 3: \"The World Cup is the most prestigious tournament in international soccer.\"\nTask: Write a similar sentence about sports.",
    "Technology": "Example 1: \"Artificial intelligence is changing the way we live and work.\"\nExample 2: \"The latest iPhone has a number of exciting new features.\"\nExample 3: \"Cybersecurity is becoming increasingly important as more and more data moves online.\"\nTask: Write a similar sentence about technology.",
    "Entertainment": "Example 1: \"The new Marvel movie is breaking box office records.\"\nExample 2: \"The Grammy Awards are a celebration of the best music of the year.\"\nExample 3: \"The latest season of Game of Thrones had fans on the edge of their seats.\"\nTask: Write a similar sentence about entertainment."
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
                max_length=100,
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
    expanded_data[category] = generate_sentences(prompt, num_sentences=5)


output_path = "cleaned_generated_train.json"
with open(output_path, 'w') as output_file:
    json.dump(expanded_data, output_file, indent=4)

print(f"Generated and cleaned data saved to {output_path}")