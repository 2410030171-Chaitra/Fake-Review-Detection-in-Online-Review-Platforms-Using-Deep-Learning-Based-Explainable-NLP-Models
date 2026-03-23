import joblib
from lime.lime_text import LimeTextExplainer

# Load model + vectorizer
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Create LIME explainer
explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

# Function for LIME
def predict_proba(texts):
    vec = vectorizer.transform(texts)
    return model.predict_proba(vec)

# Input
text = input("Enter review for explanation: ")

# Generate explanation
exp = explainer.explain_instance(
    text,
    predict_proba,
    num_features=6
)

# Show explanation in browser
exp.show_in_notebook()

# Save as HTML (VERY IMPORTANT for PPT)
exp.save_to_file("lime_explanation.html")

print("\n✅ Explanation saved as lime_explanation.html")