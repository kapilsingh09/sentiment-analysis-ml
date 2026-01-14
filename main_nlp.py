# ===============================
# Sentiment Analysis Model
# ===============================

# Import required libraries
# ===============================
# Import Required Libraries
# ===============================

# Numerical computation
import numpy as np

# Data handling and manipulation
import pandas as pd

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Text processing utilities
import string
import nltk

# Model saving and loading
import joblib

# Natural Language Processing (NLP)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Machine Learning utilities
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ===============================
# Load Dataset
# ===============================

# Load the dataset (text ; emotion)
df = pd.read_csv("train.txt", sep=';', header=None, names=['text', 'emotion'])

# Check first few rows
df.head()

# Check for missing values
df.isnull().sum()

# ===============================
# Encode Emotion Labels
# ===============================

# Convert emotion labels to numerical values
unique_emotions = df['emotion'].unique()
emotion_numbers = {}

for i, emo in enumerate(unique_emotions):
    emotion_numbers[emo] = i

df['emotion'] = df['emotion'].map(emotion_numbers)

# ===============================
# Text Preprocessing
# ===============================

# Convert text to lowercase
df['text'] = df['text'].str.lower()

# Remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['text'] = df['text'].apply(remove_punctuation)

# Remove numbers
def remove_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

df['text'] = df['text'].apply(remove_numbers)

# Remove emojis and non-ASCII characters
def remove_emojis(text):
    return ''.join([char for char in text if char.isascii()])

df['text'] = df['text'].apply(remove_emojis)

# ===============================
# Stopword Removal
# ===============================

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

df['text'] = df['text'].apply(remove_stopwords)

# ===============================
# Train-Test Split
# ===============================

X = df['text']
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ===============================
# TF-IDF Vectorization
# ===============================

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# ===============================
# Model Training (Logistic Regression)
# ===============================

lr_model = LogisticRegression(max_iter=10000)

lr_model.fit(X_train_tfidf, y_train)

# ===============================
# Model Evaluation
# ===============================

y_pred = lr_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# ===============================
# Save Model and Vectorizer
# ===============================

joblib.dump(lr_model, 'sentiment_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Your mewooo is saved successfully!")
#here i use ai too make coming main_nlp.py file
