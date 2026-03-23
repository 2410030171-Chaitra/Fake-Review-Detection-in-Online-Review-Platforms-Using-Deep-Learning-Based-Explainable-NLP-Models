import pandas as pd
import joblib
from preprocessing import clean_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("reviews.csv")

df['clean'] = df['review'].apply(clean_text)

X = df['clean']
y = df['label']

# Load model
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Transform
X_vec = vectorizer.transform(X)

# Split (same way as training)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))