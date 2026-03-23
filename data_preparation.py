import pandas as pd

# Load dataset
df = pd.read_excel(r"C:\Users\LASYA PRIYA\Downloads\pdnc real vs fake reviews\Amazon_Reviews.csv.xlsx")

# Select required columns
df = df[['Review Text', 'Rating']]

# 🔥 Convert Rating properly (handles text like "5 stars", "Rated 4 out of 5")
df['Rating'] = df['Rating'].astype(str).str.extract('(\d+)')
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Remove invalid ratings
df = df.dropna(subset=['Rating'])

# Create labels
def create_label(rating):
    if rating >= 4:
        return 1   # Genuine
    elif rating <= 2:
        return 0   # Fake
    else:
        return None

df['label'] = df['Rating'].apply(create_label)

# Remove neutral reviews (rating = 3)
df = df.dropna(subset=['label'])

# Rename columns
df = df[['Review Text', 'label']]
df.columns = ['review', 'label']

# Remove empty reviews
df = df[df['review'].notnull()]
df = df[df['review'].str.strip() != ""]

# 🔥 Balance dataset (important for accuracy)
df_fake = df[df['label'] == 0]
df_real = df[df['label'] == 1]

min_len = min(len(df_fake), len(df_real))

df = pd.concat([
    df_fake.sample(min_len, random_state=42),
    df_real.sample(min_len, random_state=42)
])

# Shuffle dataset
df = df.sample(frac=1, random_state=42)

# Save final dataset
df.to_csv("reviews.csv", index=False)

# Output
print("✅ Dataset prepared successfully!")
print(df.head())
print("Total rows:", len(df))