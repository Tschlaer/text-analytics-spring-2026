# Step 2
Q: How would I show the columns, data types, number of rows, and missing values of the dataset? Code only.

R: import pandas as pd

Load the dataset
df = pd.read_csv("hotel_reviews.csv")

Show column names
print("Columns:")
print(df.columns)

Show data types
print("\nData Types:")
print(df.dtypes)

Number of rows
print("\nNumber of Rows:")
print(len(df))

Missing values per column
print("\nMissing Values:")
print(df.isna().sum())

Q: How would I get the descriptive statistics and create a text length distribution histogram for reviews.text only? Code only.

R: import pandas as pd
import matplotlib.pyplot as plt

Descriptive statistics for review text length
df["text_length"] = df["reviews.text"].astype(str).str.len()
print(df["text_length"].describe())

Histogram of text length distribution
plt.figure(figsize=(10,5))
plt.hist(df["text_length"], bins=40, color="skyblue", edgecolor="black")
plt.title("Review Text Length Distribution")
plt.xlabel("Text Length (characters)")
plt.ylabel("Frequency")
plt.show()

Q: How would I make the histogram be downloadable to png? 

R: Save to PNG
plt.savefig("text_length_histogram.png", dpi=300, bbox_inches="tight")

### Png downloaded as white box so had to edit the plt.show() line to come after plt.savefig

# Step 3

Q:Help clean data. Drop missing values standardizing text, removing white space, handling contractions, removing excess punctuation, catching spelling errors, convert numbers to words, convert date format. 

R: import pandas as pd
import re
import string
from datetime import datetime
import contractions
from textblob import TextBlob
from num2words import num2words

Load dataset
df = pd.read_csv("hotel_reviews.csv")

DROP ROWS WITH MISSING review.text VALUES
df = df.dropna(subset=["reviews.text"]).copy()

TEXT CLEANING FUNCTIONS

def clean_text(text):
    # Convert to string
    text = str(text)

    # Expand contractions (can't -> cannot)
    text = contractions.fix(text)

    # Lowercase
    text = text.lower()

    # Remove excess punctuation (keep . , ! ?)
    text = re.sub(rf"[{re.escape(string.punctuation.replace('.', '').replace(',', '').replace('!', '').replace('?', ''))}]", " ", text)

    # Convert numbers to words (e.g., "5 stars" -> "five stars")
    text = re.sub(r"\b\d+\b", lambda x: num2words(int(x.group())), text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Correct spelling (basic TextBlob)
    text = str(TextBlob(text).correct())

    return text

APPLY CLEANING TO reviews.text
df["cleaned_text"] = df["reviews.text"].apply(clean_text)

STANDARDIZE DATE COLUMNS (EXAMPLE: reviews.date)
Convert to YYYY-MM-DD format
def standardize_date(date_value):
    try:
        return pd.to_datetime(date_value).strftime("%Y-%m-%d")
    except:
        return None

date_columns = ["reviews.date", "dateAdded", "dateUpdated"]

for col in date_columns:
    if col in df.columns:
        df[col] = df[col].apply(standardize_date)

SAVE CLEANED DATASET
df.to_csv("hotel_reviews_cleaned.csv", index=False)

Q: Apply cleaning to new dataframe rather than rewriting over current dataframe

R: import pandas as pd
import re
import string
from datetime import datetime
import contractions
from textblob import TextBlob
from num2words import num2words

Load dataset
df = pd.read_csv("hotel_reviews.csv")

CREATE NEW DATAFRAME FOR CLEANING
df_clean = df.copy()

Drop missing values only in new dataframe
df_clean = df_clean.dropna(subset=["reviews.text"]).copy()

TEXT CLEANING FUNCTIONS
def clean_text(text):
    text = str(text)

    # Expand contractions
    text = contractions.fix(text)

    # Lowercase
    text = text.lower()

    # Remove excess punctuation (keep . , ! ?)
    text = re.sub(
        rf"[{re.escape(string.punctuation.replace('.', '').replace(',', '').replace('!', '').replace('?', ''))}]",
        " ",
        text,
    )

    # Convert numbers → words
    text = re.sub(r"\b\d+\b", lambda x: num2words(int(x.group())), text)

    # Reduce extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Spell correction
    text = str(TextBlob(text).correct())

    return text

APPLY CLEANING TO NEW DATAFRAME
df_clean["cleaned_text"] = df_clean["reviews.text"].apply(clean_text)

STANDARDIZE DATE COLUMNS IN NEW DATAFRAME
def standardize_date(value):
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except:
        return None

date_columns = ["reviews.date", "dateAdded", "dateUpdated"]

for col in date_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(standardize_date)

SAVE CLEAN VERSION
df_clean.to_csv("hotel_reviews_cleaned.csv", index=False)
### Cleaning applied to entire dataset rather than just to reviews.text, made changes to focus on reviews.text dataframe
### Textblob for spelling correction was taking too long to load so it was removed
### Date formatting applied to other columns not reviews.text, removed. 

# Step 5

Q: Help tokenize using VADER in df_clean

R: # If needed, install nltk (uncomment if your environment supports shell commands)
 !pip install nltk

import pandas as pd
import numpy as np

 --- NLTK + VADER setup ---
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

 Make sure required resources are available
try:
    _ = nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

try:
    _ = nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

 --- Assume df_clean already exists and has 'cleaned_text' ---
 If not, load it:
 df_clean = pd.read_csv("hotel_reviews_cleaned.csv")

 Safety: ensure cleaned_text exists and is string
df_clean["cleaned_text"] = df_clean["cleaned_text"].astype(str)

 Optional: tokenize into words (useful if you need tokens for other tasks)
df_clean["tokens"] = df_clean["cleaned_text"].apply(lambda x: word_tokenize(x))

 Initialize VADER and score each review (VADER internally tokenizes as well)
sia = SentimentIntensityAnalyzer()
scores = df_clean["cleaned_text"].apply(sia.polarity_scores)

 Expand dict scores into separate columns
scores_df = pd.DataFrame(list(scores))
scores_df.columns = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]

 Concatenate back to df_clean
df_clean = pd.concat([df_clean.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

 (Optional) save the augmented dataset
df_clean.to_csv("hotel_reviews_vader_scored.csv", index=False)

 Quick peek
print(df_clean[["cleaned_text", "vader_neg", "vader_neu", "vader_pos", "vader_compound"]].head())

Q: Help tokenize using TextBlob in df_clean

R: from textblob import TextBlob

Ensure text is string
df_clean["cleaned_text"] = df_clean["cleaned_text"].astype(str)

Tokenize using TextBlob
df_clean["tb_tokens"] = df_clean["cleaned_text"].apply(lambda x: TextBlob(x).words)

Optional: preview
print(df_clean[["cleaned_text", "tb_tokens"]].head())

Q: Help tokenize using DistilBERT

R:
Below is a ready-to-run DistilBERT tokenization & vectorization snippet for your hotel reviews.  
It uses the `bhadresh-savani/distilbert-base-uncased-emotion` tokenizer to convert text → subword tokens → `input_ids` and `attention_mask` tensors you can feed to the model.

> If you already have a cleaned column (`df_clean["cleaned_text"]`), use it; otherwise switch to `df["reviews.text"]`.



1) Install + imports

python
If your environment supports shell commands, uncomment:
!pip install transformers torch --quiet

import pandas as pd
import torch
from transformers import AutoTokenizer


2) Load data and select text column

python
Load your CSV (uncleaned or cleaned)
df = pd.read_csv("hotel_reviews.csv")

Choose the column to tokenize
text_col = "reviews.text"          # or "cleaned_text" if you created one earlier
texts = df[text_col].fillna("").astype(str).tolist()


3) Initialize tokenizer

python
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


4) Batch tokenize → tensors (input\_ids, attention\_mask)

python
Typical settings for sequence classification
MAX_LEN = 256

encodings = tokenizer(
    texts,
    padding=True,            # pad to the longest in the batch (or set to "max_length")
    truncation=True,         # truncate longer texts
    max_length=MAX_LEN,
    return_tensors="pt"      # return PyTorch tensors
)

5) (Optional) Attach tokenized vectors back to the DataFrame (IDs as lists)

Warning: this can make your CSV large; prefer keeping tensors on disk (torch.save)
df["input_ids"] = [row.tolist() for row in input_ids]
df["attention_mask"] = [row.tolist() for row in attention_mask]

Save a tokenized artifact (parquet handles lists efficiently)
df.to_parquet("hotel_reviews_tokenized.parquet", index=False)

input_ids = encodings["input_ids"]           # shape: [N, MAX_SEQ_LEN]
attention_mask = encodings["attention_mask"] # shape: [N, MAX_SEQ_LEN]

print(input_ids.shape, attention_mask.shape)
