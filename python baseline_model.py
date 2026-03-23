import pandas as pd
from preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("reviews.csv")

# Clean text
df['clean'] = df['review'].apply(clean_text)

# Features and labels
X = df['clean']
y = df['label']

# TF-IDF (important for accuracy)
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),   # bigrams improve accuracy
    min_df=2              # remove rare words
)

X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Logistic Regression model
model = LogisticRegression(max_iter=300)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("\n✅ Logistic Regression Accuracy:", acc)

# Detailed report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, "lr_model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")

print("\n✅ Model and vectorizer saved successfully!")