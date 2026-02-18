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

# Part 3

Q:Help clean data. Drop missing values standardizing text, removing white space, handling contractions, removing excess punctuation, catching spelling errors, convert numbers to words, convert date format. 

R: import pandas as pd
import re
import string
from datetime import datetime
import contractions
from textblob import TextBlob
from num2words import num2words

----------------------------
Load dataset
----------------------------
df = pd.read_csv("hotel_reviews.csv")

---------------------------------------------------
DROP ROWS WITH MISSING review.text VALUES
---------------------------------------------------
df = df.dropna(subset=["reviews.text"]).copy()

---------------------------------------------------
TEXT CLEANING FUNCTIONS
---------------------------------------------------

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

---------------------------------------------------
APPLY CLEANING TO reviews.text
---------------------------------------------------
df["cleaned_text"] = df["reviews.text"].apply(clean_text)

---------------------------------------------------
STANDARDIZE DATE COLUMNS (EXAMPLE: reviews.date)
Convert to YYYY-MM-DD format
---------------------------------------------------
def standardize_date(date_value):
    try:
        return pd.to_datetime(date_value).strftime("%Y-%m-%d")
    except:
        return None

date_columns = ["reviews.date", "dateAdded", "dateUpdated"]

for col in date_columns:
    if col in df.columns:
        df[col] = df[col].apply(standardize_date)

---------------------------------------------------
SAVE CLEANED DATASET
---------------------------------------------------
df.to_csv("hotel_reviews_cleaned.csv", index=False)

Q: Apply cleaning to new dataframe rather than rewriting over current dataframe

R: import pandas as pd
import re
import string
from datetime import datetime
import contractions
from textblob import TextBlob
from num2words import num2words

----------------------------
Load dataset
----------------------------
df = pd.read_csv("hotel_reviews.csv")

---------------------------------------------------
CREATE NEW DATAFRAME FOR CLEANING
---------------------------------------------------
df_clean = df.copy()

Drop missing values only in new dataframe
df_clean = df_clean.dropna(subset=["reviews.text"]).copy()

---------------------------------------------------
TEXT CLEANING FUNCTIONS
---------------------------------------------------
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

---------------------------------------------------
APPLY CLEANING TO NEW DATAFRAME
---------------------------------------------------
df_clean["cleaned_text"] = df_clean["reviews.text"].apply(clean_text)

---------------------------------------------------
STANDARDIZE DATE COLUMNS IN NEW DATAFRAME
---------------------------------------------------
def standardize_date(value):
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except:
        return None

date_columns = ["reviews.date", "dateAdded", "dateUpdated"]

for col in date_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(standardize_date)

---------------------------------------------------
SAVE CLEAN VERSION
---------------------------------------------------
df_clean.to_csv("hotel_reviews_cleaned.csv", index=False)
