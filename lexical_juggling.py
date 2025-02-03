import random
import pandas as pd

#define df_new
df_new = pd.read_csv('train_submission.csv') 

# Count the occurrences of each label
label_counts_new = df_new["Label"].value_counts()

# Identify labels with fewer than 200 observations to balance them all
labels_to_augment_new = label_counts_new[label_counts_new < 200]


# Expanded word list with more diversity and common words
common_words = [
    "hello", "world", "language", "words", "sentence", "paragraph", "writing", "speak", "translate",
    "computer", "science", "programming", "internet", "learning", "education", "knowledge", "university",
    "library", "history", "culture", "communication", "expression", "literature", "author", "story",
    "conversation", "global", "different", "diverse", "understand", "explain", "meaning", "definition",
    "context", "alphabet", "vocabulary", "grammar", "syntax", "structure", "linguistics", "pronunciation",
    "accent", "dialect", "semantics", "phonetics", "morphology", "etymology", "translation", "comprehension",
    "sentence", "phrase", "idiom", "colloquial", "formal", "informal", "native", "fluent", "expression",
    "message", "verbal", "written", "spoken", "speech", "communication", "conversation", "discussion", "dialogue",
    "debate", "argument", "explanation", "statement", "description", "narrative", "storytelling", "poetry",
    "prose", "novel", "book", "article", "journal", "paper", "document", "text", "manuscript", "script",
    "tablet", "scroll", "inscription", "glyph", "symbol", "character", "letter", "word", "phrase", "clause",
    "sentence", "paragraph", "chapter", "volume", "edition", "publication", "bibliography", "citation", "reference"
]

# Function to generate more diversified sentences within length constraints
def generate_diversified_sentence(min_len=40, max_len=128):
    while True:
        sentence_length = random.randint(8, 25)  # Increase word count variation
        sentence = " ".join(random.choices(common_words, k=sentence_length)).capitalize() + "."
        if min_len <= len(sentence) <= max_len:
            return sentence

new_data_new_diversified = []
new_id_new = df_new["ID"].max() + 1 if not df_new["ID"].isnull().all() else 1

for label, missing_count in labels_to_augment_new.items():
    for _ in range(200 - missing_count):
        new_data_new_diversified.append([new_id_new, "Public", generate_diversified_sentence(), label])
        new_id_new += 1

new_data_df_new_diversified = pd.DataFrame(new_data_new_diversified, columns=["ID", "Usage", "Text", "Label"])
balanced_df_new_diversified = pd.concat([df_new[["ID", "Usage", "Text", "Label"]], new_data_df_new_diversified], ignore_index=True)

# Save the enriched dataset with more diversified text
balanced_file_path_new_diversified = "train_submission_balanced_diversified.csv"
balanced_df_new_diversified.csv(balanced_file_path_new_diversified, index=False)
