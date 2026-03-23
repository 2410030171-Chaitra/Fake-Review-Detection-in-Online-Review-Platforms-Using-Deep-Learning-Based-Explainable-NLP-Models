import streamlit as st
import joblib
from preprocessing import clean_text
from lime.lime_text import LimeTextExplainer

# Load model
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

def predict_proba(texts):
    vec = vectorizer.transform(texts)
    return model.predict_proba(vec)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fake Review Detection", layout="wide")

# ---------------- GLOBAL STYLES ----------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Main container spacing */
.block-container {
    padding-top: 4rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: #00e5ff;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #b0bec5;
    margin-bottom: 30px;
}

/* Textarea */
textarea {
    border-radius: 12px !important;
    background-color: #1e2a38 !important;
    color: white !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    border: none;
}

/* Result */
.result {
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
    margin-top: 10px;
}

.genuine {
    background: rgba(0,255,150,0.15);
    border-left: 5px solid #00e676;
}

.fake {
    background: rgba(255,80,80,0.15);
    border-left: 5px solid #ff5252;
}

/* Tags */
.tag {
    display: inline-block;
    padding: 8px 14px;
    margin: 6px;
    border-radius: 20px;
    background: rgba(0,200,255,0.2);
    color: #00e5ff;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">Fake Review Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered system to classify reviews with explainable insights</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
review = st.text_area("Enter Review", height=160, placeholder="Type or paste a product review here...")

# ---------------- ACTION ----------------
if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review")

    else:
        # Prediction
        clean = clean_text(review)
        vec = vectorizer.transform([clean])
        prob = model.predict_proba(vec)[0][1]
        label = "Genuine Review" if prob > 0.5 else "Fake Review"

        # LIME
        exp = explainer.explain_instance(review, predict_proba, num_features=6)
        words = [w[0] for w in exp.as_list()]

        # ---------------- OUTPUT ----------------
        st.markdown("## Result")

        if label == "Genuine Review":
            st.markdown(f'<div class="result genuine">✔ {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result fake">✖ {label}</div>', unsafe_allow_html=True)

        st.write(f"Confidence Score: **{round(prob, 2)}**")

        # Progress bar
        st.progress(prob)

        # Influencing words
        st.markdown("## Key Influencing Words")

        tag_html = ""
        for w in words:
            tag_html += f'<span class="tag">{w}</span>'

        st.markdown(tag_html, unsafe_allow_html=True)