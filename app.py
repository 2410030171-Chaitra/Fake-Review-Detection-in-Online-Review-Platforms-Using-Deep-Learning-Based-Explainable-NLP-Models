import joblib
from preprocessing import clean_text

# Load model
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

def predict(review):
    clean = clean_text(review)
    vec = vectorizer.transform([clean])
    prob = model.predict_proba(vec)[0][1]

    label = "Genuine Review" if prob > 0.5 else "Fake Review"

    return label, prob

# Input
review = input("Enter Review: ")

label, confidence = predict(review)

print("\nPrediction:", label)
print("Confidence Score:", round(confidence, 2))