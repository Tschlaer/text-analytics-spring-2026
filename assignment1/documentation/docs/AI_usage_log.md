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
