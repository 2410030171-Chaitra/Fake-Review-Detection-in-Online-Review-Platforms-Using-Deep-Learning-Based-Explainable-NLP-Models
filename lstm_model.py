import pandas as pd
from preprocessing import clean_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load data
df = pd.read_csv("reviews.csv")

df['clean'] = df['review'].apply(clean_text)

texts = df['clean']
labels = df['label']

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(sequences, maxlen=150)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=150),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)

print("\n✅ LSTM Accuracy:", acc)

# Save
model.save("lstm_model.keras")
tf.keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)