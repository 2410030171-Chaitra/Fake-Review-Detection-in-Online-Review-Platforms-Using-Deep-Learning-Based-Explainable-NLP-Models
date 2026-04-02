import re
import nltk
from nltk.corpus import stopwords

# Download required data safely
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # remove symbols
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
