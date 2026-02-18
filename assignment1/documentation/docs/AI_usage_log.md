# Step 2
Q: How would I show the columns, data types, number of rows, and missing values of the dataset? Code only.
R: 
import pandas as pd

# Load the dataset
df = pd.read_csv("hotel_reviews.csv")

# Show column names
print("Columns:")
print(df.columns)

# Show data types
print("\nData Types:")
print(df.dtypes)

# Number of rows
print("\nNumber of Rows:")
print(len(df))

# Missing values per column
print("\nMissing Values:")
print(df.isna().sum())

