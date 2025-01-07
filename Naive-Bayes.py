import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews, stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import string

# nltk.download('movie_reviews')
# nltk.download('stopwords')

documents = [
    (" ".join(movie_reviews.words(fileid)),category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

stop_words = set(stopwords.words('english'))

df = pd.DataFrame(documents, columns=['review','sentiment'])

print(df.head())

# Clean text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    text = text.split()
    filtered_words = [word for word in text if word not in stop_words]
    filtered_text = " ".join(filtered_words)
    return filtered_text

df['cleantext'] = df['review'].apply(preprocess_text)
print(df['review'].head())
print(df['cleantext'].head())

# Convert text to vectors
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tain Naive Bayes CLassifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"accuracy: {accuracy_score(y_test, y_pred)}")
print(f"classification report: \n{classification_report(y_test, y_pred)}")

# prediction
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

# test prediction
text_sampel = ["This book is so good and best", "I hate my self and I want to kill myself", "I love my jobs because it is very fun"]
for t in text_sampel:
    print(predict_sentiment(t))